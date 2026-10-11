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
SERIAL_PORT = "/dev/ttyAMA0"      # 串口设备路径
BAUDRATE = 57600                  # 波特率
TRIGGER_PIN_1 = 26                # GPIO触发引脚1 (BCM编号)
TRIGGER_PIN_2 = 16                # GPIO触发引脚2 (BCM编号)
TRIGGER_DURATION = 2.0            # 中心点停留触发时长(秒)
OUTPUT_DURATION = 1.5             # GPIO输出持续时间(秒)
HEIGHT_PIANYI = -1                 # height_pianyi(xiangsu)
WEIGHT_PIANYI = 35                # shuiping_pianyi(xiangsu)

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
        self.tracking_timeout = 0.010  # 跟踪超时时间(秒)

        # ----- 控制状态 -----
        self.center_in_box_start_time = None
        self.output_start_time_1 = None
        self.output_start_time_2 = None
        self.last_sent_time = 0
        self.SEND_INTERVAL = 0.02    # 串口发送间隔(秒)

    def _init_tracker(self, bbox):
        """初始化CSRT跟踪器"""
        self.tracker = cv2.legacy.TrackerCSRT_create()
        x, y, w, h = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
        dummy_frame = np.zeros((720,1280,3), dtype=np.uint8)  # 虚拟图像
        self.tracker.init(dummy_frame, (int(x), int(y), int(w), int(h)))
        self.last_detection_time = time.time()
        print(f"[TRACKER] Initialized with bbox: {bbox}")

    def _update_tracker(self):
        """更新跟踪器状态（带完整异常处理）"""
        if self.tracker or bbox is None:
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
            if (w <= 0 or h <= 0 or                  # 尺寸无效
                x >= 1280 or y >= 720 or              # 超出右/下边界
                (x + w) <= 0 or (y + h) <= 0):      # 超出左/上边界
                print(f"[TRACKER] Invalid bbox: {x},{y},{w},{h}")
                return None
                
            # 返回[x1,y1,x2,y2]格式（确保在图像范围内）
            return [
                max(0, min(x, 1279)),
                max(0, min(y, 719)),
                max(0, min(x + w, 1279)),
                max(0, min(y + h, 719))
            ]
            
        except Exception as e:
            print(f"[TRACKER ERROR] {str(e)}")
            return None

    def _is_center_in_box(self, box, width, height):
        """检查中心点是否在边界框内（向左偏移5像素）"""
        center_x = (width / 2) + WEIGHT_PIANYI  # 关键修改点
        center_y = height / 2 + HEIGHT_PIANYI
        return (box[0] <= center_x <= box[2]) and (box[1] <= center_y <= box[3])

    def _update_gpio_state(self):
        """GPIO状态管理"""
        current_time = time.time()

        # GPIO1超时检查
        if self.output_start_time_1 and (current_time - self.output_start_time_1 >= OUTPUT_DURATION):
            GPIO.output(TRIGGER_PIN_1, GPIO.LOW)
            self.output_start_time_1 = None
            print("[GPIO1] Turned OFF by timeout")

        # GPIO2超时检查
        if self.output_start_time_2 and (current_time - self.output_start_time_2 >= OUTPUT_DURATION):
            GPIO.output(TRIGGER_PIN_2, GPIO.LOW)
            self.output_start_time_2 = None
            print("[GPIO2] Turned OFF by timeout")

        # 触发条件检查
        if self.center_in_box_start_time and \
           (current_time - self.center_in_box_start_time >= TRIGGER_DURATION):
            if not self.output_start_time_1:
                GPIO.output(TRIGGER_PIN_1, GPIO.HIGH)
                self.output_start_time_1 = current_time
                print("[GPIO1] Turned ON by trigger")
            if not self.output_start_time_2:
                GPIO.output(TRIGGER_PIN_2, GPIO.HIGH)
                self.output_start_time_2 = current_time
                print("[GPIO2] Turned ON by trigger")

    def _send_data(self, box, width, height):
        """发送目标中心点相对于图像中心的像素偏移量（格式：&X,Y#）
        示例: 
          &-50.0,30.0#  # 目标在中心左侧50像素、上方30像素
        """
        if not self.ser or box is None:
            return

        # 计算目标中心点坐标
        target_center_x = (box[0] + box[2]) * 0.5
        target_center_y = (box[1] + box[3]) * 0.5

        # 计算相对偏移（限制在±图像尺寸内）
        offset_x = min(max((width / 2)-target_center_x, -width/2), width/2)
        offset_y = min(max(target_center_y-(height / 2) , -height/2), height/2)

        # 构建协议字符串
        data_str = f"&{offset_x:.1f},{offset_y:.1f}#"
        
        try:
            self.ser.write(data_str.encode('ascii'))
            print(f"[SERIAL] Sent: {data_str}")
        except Exception as e:
            print(f"[SERIAL ERROR] {e}")

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

    # 阶段1: YOLO检测
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    if detections:
        # 选择面积最大的检测框
        largest_box = max(
            detections,
            key=lambda d: (d.get_bbox().xmax()-d.get_bbox().xmin())*(d.get_bbox().ymax()-d.get_bbox().ymin())
        )
        bbox = largest_box.get_bbox()
        active_bbox = [
            bbox.xmin() * width,
            bbox.ymin() * height,
            bbox.xmax() * width,
            bbox.ymax() * height
        ]
        user_data.last_detection_time = current_time
        if not user_data.tracker:
            user_data._init_tracker(active_bbox)
        else:
            dummy_frame = np.zeros((height,width,3), dtype=np.uint8)
            user_data.tracker.init(dummy_frame, (
                int(active_bbox[0]), int(active_bbox[1]),
                int(active_bbox[2]-active_bbox[0]),
                int(active_bbox[3]-active_bbox[1])
            ))
    
    # 阶段2: 跟踪器补偿
    elif user_data.tracker and (current_time - user_data.last_detection_time < user_data.tracking_timeout):
        active_bbox = user_data._update_tracker()

    # 阶段3: 控制逻辑
    if active_bbox:
        # 中心点检测（使用偏移后的中心）
        if user_data._is_center_in_box(active_bbox, width, height):
            if not user_data.center_in_box_start_time:
                user_data.center_in_box_start_time = current_time
                print("[LOGIC] Center entered trigger zone")
        else:
            user_data.center_in_box_start_time = None
        
        # 节流发送串口数据
        if current_time - user_data.last_sent_time > user_data.SEND_INTERVAL:
            user_data._send_data(active_bbox, width, height)
            user_data.last_sent_time = current_time
        
        # 更新GPIO状态
        user_data._update_gpio_state()
    else:
        user_data.center_in_box_start_time = None
        user_data._update_gpio_state()

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
        print("System shutdown complete")
