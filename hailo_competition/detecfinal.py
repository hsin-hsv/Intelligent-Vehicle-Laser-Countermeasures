import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import time
import serial
import RPi.GPIO as GPIO

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

# Serial and GPIO configuration
SERIAL_PORT = "/dev/ttyAMA0"
BAUDRATE = 57600
TRIGGER_PIN_1 = 26  # BCM26 pin
TRIGGER_PIN_2 = 16  # BCM16 pin
TRIGGER_DURATION = 2.0  # Trigger duration in seconds
OUTPUT_DURATION = 1.5  # Output duration in seconds

class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        # Initialize GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(TRIGGER_PIN_1, GPIO.OUT)
        GPIO.setup(TRIGGER_PIN_2, GPIO.OUT)
        GPIO.output(TRIGGER_PIN_1, GPIO.LOW)
        GPIO.output(TRIGGER_PIN_2, GPIO.LOW)
        
        # Initialize serial port
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
            print(f"[SERIAL] Port {SERIAL_PORT} opened successfully")
        except Exception as e:
            print(f"[SERIAL ERROR] Port initialization failed: {e}")
            self.ser = None
        
        # State variables
        self.center_in_box_start_time = None
        self.output_start_time_1 = None  # For BCM26
        self.output_start_time_2 = None  # For BCM16
        self.last_sent_time = 0
        self.SEND_INTERVAL = 0.1  # Data sending interval in seconds
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()

    def _is_center_in_box(self, bbox, width, height):
        """Check if object center is within central area of frame"""
        xmin = bbox.xmin()
        xmax = bbox.xmax()
        ymin = bbox.ymin()
        ymax = bbox.ymax()
        center_x = width / 2
        center_y = height / 2
        return (xmin <= center_x <= xmax) and (ymin <= center_y <= ymax)

    def _update_gpio_state(self):
        """Update GPIO output states based on detection"""
        current_time = time.time()

        # Handle BCM26 (TRIGGER_PIN_1)
        if self.output_start_time_1 is not None:
            if current_time - self.output_start_time_1 >= OUTPUT_DURATION:
                GPIO.output(TRIGGER_PIN_1, GPIO.LOW)
                self.output_start_time_1 = None
                print("[GPIO] Output 1 (BCM26) turned OFF")

        # Handle BCM16 (TRIGGER_PIN_2)
        if self.output_start_time_2 is not None:
            if current_time - self.output_start_time_2 >= OUTPUT_DURATION:
                GPIO.output(TRIGGER_PIN_2, GPIO.LOW)
                self.output_start_time_2 = None
                print("[GPIO] Output 2 (BCM16) turned OFF")

        # Activate outputs if object has been centered long enough
        if self.center_in_box_start_time is not None:
            if current_time - self.center_in_box_start_time >= TRIGGER_DURATION:
                if self.output_start_time_1 is None:
                    GPIO.output(TRIGGER_PIN_1, GPIO.HIGH)
                    self.output_start_time_1 = current_time
                    print("[GPIO] Output 1 (BCM26) turned ON")
                if self.output_start_time_2 is None:
                    GPIO.output(TRIGGER_PIN_2, GPIO.HIGH)
                    self.output_start_time_2 = current_time
                    print("[GPIO] Output 2 (BCM16) turned ON")

    def _send_data(self, bbox, width, height, class_name):
        """Send object data via serial port"""
        if not self.ser:
            return

        # Get bounding box coordinates
        xmin = bbox.xmin()
        xmax = bbox.xmax()
        ymin = bbox.ymin()
        ymax = bbox.ymax()

        # Calculate object center coordinates (relative to frame center)
        center = np.array([
            (xmin + xmax) * 0.5 - width * 0.5,
            (ymin + ymax) * 0.5 - height * 0.5
        ])

        # Prepare data packet: [start_byte, center_x, center_y, class_id, gpio_states]
        data = bytearray(14)
        data[0] = 0x24  # Start byte
        data[1:5] = np.float32(center[0]).tobytes()
        data[5:9] = np.float32(center[1]).tobytes()
        # Encode class name as 4-byte float (simple hash)
        class_hash = float(abs(hash(class_name))) % 10000
        data[9:13] = np.float32(class_hash).tobytes()
        # Add GPIO states (1 byte)
        gpio_state = (1 if self.output_start_time_1 is not None else 0) | \
                     (2 if self.output_start_time_2 is not None else 0)
        data[13] = gpio_state

        try:
            self.ser.write(data)
            print(f"[SERIAL] Sent object data - Class: {class_name}, Center: {center}, GPIOs: {gpio_state:02b}")
        except Exception as e:
            print(f"[SERIAL ERROR] Data transmission failed: {e}")

def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Update frame counter and calculate FPS
    user_data.frame_count += 1
    current_time = time.time()
    if current_time - user_data.last_fps_time >= 1.0:
        user_data.fps = user_data.frame_count / (current_time - user_data.last_fps_time)
        user_data.frame_count = 0
        user_data.last_fps_time = current_time
        print(f"[PERF] Current FPS: {user_data.fps:.1f}")

    # Get video frame parameters
    format, width, height = get_caps_from_pad(pad)

    # Get video frame
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get detection results
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Process detections
    detection_count = 0
    primary_detection = None
    largest_area = 0
    
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        
        # Calculate detection box area using proper bbox methods
        width_bbox = bbox.xmax() - bbox.xmin()
        height_bbox = bbox.ymax() - bbox.ymin()
        area = width_bbox * height_bbox
        
        # Select primary detection (largest area or highest confidence)
        if area > largest_area or (area == largest_area and confidence > primary_detection['confidence'] if primary_detection else False):
            largest_area = area
            primary_detection = {
                'bbox': bbox,
                'label': label,
                'confidence': confidence,
                'area': area
            }
        
        detection_count += 1

    # If any object detected
    if primary_detection is not None:
        # Update center position status
        if user_data._is_center_in_box(primary_detection['bbox'], width, height):
            if user_data.center_in_box_start_time is None:
                user_data.center_in_box_start_time = time.time()
                print(f"[DETECTION] {primary_detection['label']} entered center area")
        else:
            if user_data.center_in_box_start_time is not None:
                print(f"[DETECTION] {primary_detection['label']} left center area")
            user_data.center_in_box_start_time = None
        
        # Send data periodically
        if time.time() - user_data.last_sent_time > user_data.SEND_INTERVAL:
            user_data._send_data(primary_detection['bbox'], width, height, primary_detection['label'])
            user_data.last_sent_time = time.time()
        
        # Update GPIO states
        user_data._update_gpio_state()
    else:
        if user_data.center_in_box_start_time is not None:
            print("[DETECTION] No objects detected")
        user_data.center_in_box_start_time = None

    # Draw information on frame
    if user_data.use_frame and frame is not None:
        # Draw detection box
        if primary_detection is not None:
            bbox = primary_detection['bbox']
            x1, y1 = int(bbox.xmin() * width), int(bbox.ymin() * height)
            x2, y2 = int(bbox.xmax() * width), int(bbox.ymax() * height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{primary_detection['label']}: {primary_detection['confidence']:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw frame center point
        center_x = width // 2
        center_y = height // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Draw system information
        cv2.putText(frame, f"FPS: {user_data.fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        gpio_state_1 = "ON" if user_data.output_start_time_1 is not None else "OFF"
        gpio_state_2 = "ON" if user_data.output_start_time_2 is not None else "OFF"
        cv2.putText(frame, f"GPIO26: {gpio_state_1}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"GPIO16: {gpio_state_2}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Convert to BGR format and store
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    try:
        app.run()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        if user_data.ser:
            user_data.ser.close()
        GPIO.cleanup()
        print("Resources released")
