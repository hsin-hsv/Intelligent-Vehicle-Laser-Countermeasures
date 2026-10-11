import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import cv2
import hailo
import time
import serial
import RPi.GPIO as GPIO

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

# ===== 硬件配置 =====
SERIAL_PORT = "/dev/ttyAMA0"  # 串口设备路径
BAUDRATE = 57600  # 波特率
TRIGGER_PIN_1 = 26  # GPIO触发引脚1 (BCM编号)
TRIGGER_PIN_2 = 16  # GPIO触发引脚2 (BCM编号)
TRIGGER_DURATION = 2.0  # 中心点停留触发时长(秒)
OUTPUT_DURATION = 1.5  # GPIO输出持续时间(秒)


class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        # ----- 硬件初始化 -----
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(TRIGGER_PIN_1, GPIO.OUT)
        GPIO.setup(TRIGGER_PIN_2, GPIO.OUT)
        GPIO.output(TRIGGER_PIN_1, GPIO.LOW)
        GPIO.output(TRIGGER_PIN_2, GPIO.LOW)

        # ----- 串口初始化 -----
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
            print(f"[SERIAL] Port {SERIAL_PORT} opened at {BAUDRATE} baud")
        except Exception as e:
            print(f"[SERIAL ERROR] {e}")
            self.ser = None

        # ----- 跟踪器配置 -----
        self.tracker = None
        self.last_detection_time = 0
        self.tracking_timeout = 0.15  # 跟踪超时时间(秒)

        # ----- 控制状态 -----
        self.center_in_box_start_time = None
        self.output_start_time_1 = None
        self.output_start_time_2 = None
        self.last_sent_time = 0
        self.SEND_INTERVAL = 0.05  # 串口发送间隔(秒)

        # ----- 可视化设置 -----
        self.debug_window_name = "Object Tracking"
        cv2.namedWindow(self.debug_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.debug_window_name, 800, 600)
        self.last_visualization_time = 0
        self.visualization_interval = 0.1  # 显示间隔(秒)

    def _init_tracker(self, bbox):
        """初始化CSRT跟踪器"""
        self.tracker = cv2.legacy.TrackerCSRT_create()
        x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]

        # 确保尺寸有效
        w = max(w, 5)
        h = max(h, 5)

        # 创建带噪声的虚拟帧
        dummy_frame = np.random.randint(0, 30, (720, 1280, 3), dtype=np.uint8)
        self.tracker.init(dummy_frame, (int(x), int(y), int(w), int(h)))
        self.last_detection_time = time.time()
        print(f"[TRACKER] Initialized with bbox: x={x}, y={y}, w={w}, h={h}")

    def _update_tracker(self):
        """更新跟踪器状态（带完整异常处理）"""
        if not self.tracker:  # 修复条件：检查跟踪器是否存在
            return None

        try:
            # 使用带噪声的虚拟帧（避免跟踪全黑画面）
            dummy_frame = np.random.randint(0, 30, (720, 1280, 3), dtype=np.uint8)

            # 更新跟踪器
            success, bbox = self.tracker.update(dummy_frame)

            # 验证跟踪结果
            if not success or bbox is None:
                print("[TRACKER] Tracking failed or lost target")
                return None

            # 解构bbox值并转换为整数
            x, y, w, h = map(int, bbox)

            # 边界检查（防止出现无效坐标）
            x1 = max(0, min(x, 1279))
            y1 = max(0, min(y, 719))
            w = max(5, min(w, 1280 - x1))
            h = max(5, min(h, 720 - y1))
            x2 = min(x1 + w, 1279)
            y2 = min(y1 + h, 719)

            print(f"[TRACKER] Updated bbox: x={x1}, y={y1}, w={w}, h={h}")
            return [x1, y1, x2, y2]

        except Exception as e:
            print(f"[TRACKER ERROR] {str(e)}")
            return None

    def _is_center_in_box(self, box, width, height):
        """检查中心点是否在边界框内（向左偏移35像素）"""
        center_x = (width / 2) + 35  # 关键修改点：向左偏移
        center_y = height / 2
        return (box[0] <= center_x <= box[2]) and (box[1] <= center_y <= box[3])

    def _update_gpio_state(self):
        """GPIO状态管理（修复逻辑错误）"""
        current_time = time.time()

        # 检查中心点停留是否超时
        should_trigger = (
                self.center_in_box_start_time is not None and
                (current_time - self.center_in_box_start_time >= TRIGGER_DURATION)
        )

        # GPIO1控制逻辑
        if should_trigger:
            if not self.output_start_time_1:
                print("[GPIO1] Turned ON")
                GPIO.output(TRIGGER_PIN_1, GPIO.HIGH)
                self.output_start_time_1 = current_time
        elif self.output_start_time_1:  # 不在触发状态但GPIO开启
            if current_time - self.output_start_time_1 >= OUTPUT_DURATION:
                print("[GPIO1] Turned OFF")
                GPIO.output(TRIGGER_PIN_1, GPIO.LOW)
                self.output_start_time_1 = None

        # GPIO2控制逻辑（与GPIO1独立）
        if should_trigger:
            if not self.output_start_time_2:
                print("[GPIO2] Turned ON")
                GPIO.output(TRIGGER_PIN_2, GPIO.HIGH)
                self.output_start_time_2 = current_time
        elif self.output_start_time_2:  # 不在触发状态但GPIO开启
            if current_time - self.output_start_time_2 >= OUTPUT_DURATION:
                print("[GPIO2] Turned OFF")
                GPIO.output(TRIGGER_PIN_2, GPIO.LOW)
                self.output_start_time_2 = None

    def _send_data(self, box, width, height):
        """发送目标中心点相对于图像中心的像素偏移量"""
        if not self.ser or box is None:
            return

        # 计算目标中心点坐标
        target_center_x = (box[0] + box[2]) * 0.5
        target_center_y = (box[1] + box[3]) * 0.5

        # 计算相对偏移（限制在±图像尺寸内）
        offset_x = min(max(target_center_x - (width / 2 + 35), -width / 2), width / 2)
        offset_y = min(max(target_center_y - (height / 2), -height / 2), height / 2)

        # 构建协议字符串
        data_str = f"&{offset_x:.1f},{offset_y:.1f}#"

        try:
            self.ser.write(data_str.encode('ascii'))
            print(f"[SERIAL] Sent: {data_str}")
        except Exception as e:
            print(f"[SERIAL ERROR] {e}")

    def _visualize_frame(self, width, height, active_bbox, trigger_status):
        """在视频帧上绘制中心点和跟踪信息"""
        # 创建虚拟帧用于可视化
        frame = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)

        # 绘制真实中心点（白色）
        real_center = (int(width // 2), int(height // 2))
        cv2.circle(frame, real_center, 8, (255, 255, 255), -1)
        cv2.circle(frame, real_center, 10, (200, 200, 200), 2)

        # 绘制偏移中心点（触发区域）
        offset_center = (real_center[0] - 35, real_center[1])  # 向左35像素
        color = (0, 0, 255) if trigger_status else (100, 100, 255)  # 红色表示激活，浅红表示未激活

        # 绘制带十字线的偏移中心点
        cv2.circle(frame, offset_center, 8, color, -1)
        cv2.line(frame, (offset_center[0] - 15, offset_center[1]),
                 (offset_center[0] + 15, offset_center[1]), color, 2)
        cv2.line(frame, (offset_center[0], offset_center[1] - 15),
                 (offset_center[0], offset_center[1] + 15), color, 2)

        # 绘制偏移指示线
        cv2.line(frame, real_center, offset_center, (0, 0, 150), 2)

        # 如果有检测目标，绘制边界框和中心点
        if active_bbox is not None:
            # 解构边界框坐标
            x1, y1, x2, y2 = map(int, active_bbox)

            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制目标中心点
            target_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(frame, target_center, 6, (0, 255, 0), -1)

            # 绘制目标到偏移中心的连线
            cv2.line(frame, target_center, offset_center, (0, 200, 0), 2)

        # 添加状态文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        status = "ACTIVE" if trigger_status else "INACTIVE"
        status_color = (0, 0, 255) if trigger_status else (200, 200, 0)
        cv2.putText(frame, f"Trigger Status: {status}", (10, 30),
                    font, 0.7, status_color, 2)

        # 显示帧
        cv2.imshow(self.debug_window_name, frame)
        cv2.waitKey(1)  # 等待1ms以更新窗口


def app_callback(pad, info, user_data):
    """GStreamer帧回调函数"""
    buffer = info.get_buffer()
    if not buffer:
        return Gst.PadProbeReturn.OK

    # 获取帧信息
    format, width, height = get_caps_from_pad(pad)
    if None in (format, width, height):
        return Gst.PadProbeReturn.OK

    # ===== 主检测流程 =====
    current_time = time.time()
    active_bbox = None
    trigger_status = False  # 中心点是否在触发区域

    # 阶段1: YOLO检测
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    if detections:
        # 选择面积最大的检测框
        largest_box = max(
            detections,
            key=lambda d: (d.get_bbox().xmax() - d.get_bbox().xmin()) * (d.get_bbox().ymax() - d.get_bbox().ymin())
        )
        bbox = largest_box.get_bbox()
        active_bbox = [
            bbox.xmin() * width,
            bbox.ymin() * height,
            bbox.xmax() * width,
            bbox.ymax() * height
        ]
        user_data.last_detection_time = current_time

        # 重新初始化跟踪器
        try:
            if user_data.tracker:
                user_data.tracker = None  # 先释放旧跟踪器
            user_data._init_tracker(active_bbox)
        except Exception as e:
            print(f"[TRACKER INIT ERROR] {e}")
            user_data.tracker = None

    # 阶段2: 跟踪器补偿
    elif user_data.tracker and (current_time - user_data.last_detection_time < user_data.tracking_timeout):
        active_bbox = user_data._update_tracker()

    # 阶段3: 控制逻辑
    if active_bbox:
        # 中心点检测（使用偏移后的中心）
        if user_data._is_center_in_box(active_bbox, width, height):
            if user_data.center_in_box_start_time is None:
                user_data.center_in_box_start_time = current_time
                print("[LOGIC] Center entered trigger zone")
            trigger_status = True
        else:
            user_data.center_in_box_start_time = None
            trigger_status = False

        # 节流发送串口数据
        if current_time - user_data.last_sent_time > user_data.SEND_INTERVAL:
            user_data._send_data(active_bbox, width, height)
            user_data.last_sent_time = current_time
    else:
        user_data.center_in_box_start_time = None
        trigger_status = False

    # 更新GPIO状态
    user_data._update_gpio_state()

    # 可视化帧状态
    if current_time - user_data.last_visualization_time > user_data.visualization_interval:
        user_data._visualize_frame(width, height, active_bbox, trigger_status)
        user_data.last_visualization_time = current_time

    return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nUser interrupted")
    finally:
        if user_data.ser:
            user_data.ser.close()
        GPIO.cleanup()
        cv2.destroyAllWindows()
        print("System shutdown complete")