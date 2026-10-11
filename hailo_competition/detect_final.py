import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import cv2
import hailo
import time
import serial
import RPi.GPIO as GPIO
from hailo_platform import PcieDevice
from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp
import json
import argparse

# Configuration parameters
SERIAL_PORT = "/dev/ttyAMA0"
BAUDRATE = 57600
SEND_INTERVAL = 0.1  # seconds
TRIGGER_PIN_1 = 26  # BCM26 pin
TRIGGER_PIN_2 = 16  # BCM16 pin
TRIGGER_DURATION = 2.0  # seconds
OUTPUT_DURATION = 1.5  # seconds
CONFIDENCE_THRESHOLD = 0.5  # Minimum detection confidence

class user_app_callback_class(app_callback_class):
    def __init__(self, labels_dict=None):
        super().__init__()
        # Hardware control
        self._init_gpio()
        self._init_serial()
        # State variables
        self.center_in_box_start_time = None
        self.output_start_time = None
        self.last_sent = 0
        self.labels_dict = labels_dict or {}
        # Fixed detection size (compatible with Hailo models)
        self.detect_size = (640, 640)

    def _init_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(TRIGGER_PIN_1, GPIO.OUT)
        GPIO.setup(TRIGGER_PIN_2, GPIO.OUT)
        GPIO.output(TRIGGER_PIN_1, GPIO.LOW)
        GPIO.output(TRIGGER_PIN_2, GPIO.LOW)

    def _init_serial(self):
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
            print(f"[SER] Port {SERIAL_PORT} opened")
        except Exception as e:
            print(f"[SER] Error: {e}")

    def _is_center_in_box(self, box):
        center_x = self.detect_size[0] / 2
        center_y = self.detect_size[1] / 2
        return (box[0] <= center_x <= box[2]) and (box[1] <= center_y <= box[3])

    def _update_gpio_state(self):
        current_time = time.time()
        if self.output_start_time is not None:
            if current_time - self.output_start_time >= OUTPUT_DURATION:
                GPIO.output(TRIGGER_PIN_1, GPIO.LOW)
                GPIO.output(TRIGGER_PIN_2, GPIO.LOW)
                self.output_start_time = None
        if self.center_in_box_start_time is not None:
            if current_time - self.center_in_box_start_time >= TRIGGER_DURATION:
                if self.output_start_time is None:
                    GPIO.output(TRIGGER_PIN_1, GPIO.HIGH)
                    GPIO.output(TRIGGER_PIN_2, GPIO.HIGH)
                    self.output_start_time = current_time

    def _send_data(self, box, label):
        if not self.ser:
            return
        center_x = (box[0] + box[2]) * 0.5 - self.detect_size[0] * 0.5
        center_y = (box[1] + box[3]) * 0.5 - self.detect_size[1] * 0.5
        data_str = f"&{label}:{center_x:.2f},{center_y:.2f}#\n"
        self.ser.write(data_str.encode('utf-8'))
        print(f"[SER] Sent: {data_str.strip()}")

    def __del__(self):
        if hasattr(self, 'ser') and self.ser:
            self.ser.close()
        GPIO.cleanup()

def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Get frame from buffer
    format, width, height = get_caps_from_pad(pad)
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get and process detections
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    best_box = None
    best_label = ""
    max_area = 0
    
    for detection in detections:
        label_id = detection.get_label_id()
        label = user_data.labels_dict.get(str(label_id), f"class_{label_id}")
        confidence = detection.get_confidence()
        
        if confidence >= CONFIDENCE_THRESHOLD:
            bbox = detection.get_bbox()
            box = [bbox.xmin(), bbox.ymin(), bbox.xmax(), bbox.ymax()]
            area = (box[2] - box[0]) * (box[3] - box[1])
            
            if area > max_area:
                max_area = area
                best_box = box
                best_label = label
            
            # Track ID handling (if available)
            track_id = ""
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = f" ID:{track[0].get_id()}"
            
            string_to_print += f"Detection:{track_id} {label} Conf:{confidence:.2f} Box:{box}\n"

    # Center trigger and serial communication
    if best_box is not None:
        if user_data._is_center_in_box(best_box):
            if user_data.center_in_box_start_time is None:
                user_data.center_in_box_start_time = time.time()
        else:
            user_data.center_in_box_start_time = None

        if time.time() - user_data.last_sent > SEND_INTERVAL:
            user_data._send_data(best_box, best_label)
            user_data.last_sent = time.time()

    user_data._update_gpio_state()

    if user_data.use_frame and frame is not None:
        if best_box is not None:
            # Draw bounding box
            cv2.rectangle(frame, 
                         (int(best_box[0]), int(best_box[1])),
                         (int(best_box[2]), int(best_box[3])),
                         (0, 255, 0), 2)
            # Draw label
            cv2.putText(frame, f"{best_label}", 
                       (int(best_box[0]), int(best_box[1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hef-path', type=str, required=True, help='Path to HEF file')
    parser.add_argument('--input', type=str, default="usb", help='Input source (e.g. usb, 0)')
    parser.add_argument('--labels-json', type=str, required=True, help='Path to labels JSON file')
    args = parser.parse_args()

    # Load labels
    try:
        with open(args.labels_json) as f:
            labels_dict = json.load(f)
        print(f"Loaded {len(labels_dict)} labels from {args.labels_json}")
    except Exception as e:
        print(f"Error loading labels: {e}")
        exit(1)

    # Initialize PCIe device
    try:
        devices = PcieDevice.scan_devices()
        if not devices:
            raise RuntimeError("No Hailo devices found")
    except Exception as e:
        print(f"Device init error: {e}")
        exit(1)

    user_data = user_app_callback_class(labels_dict=labels_dict)
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()