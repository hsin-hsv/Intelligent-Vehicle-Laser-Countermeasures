#!/bin/bash
source /home/junyan/hailo/hailo-rpi5-examples/venv_hailo_rpi5_examples/bin/activate
source /home/junyan/hailo/hailo-rpi5-examples/setup_env.sh
python3 /home/junyan/hailo/hailo-rpi5-examples/test.py --hef-path /home/junyan/hailo/hailo-rpi5-examples/resources/yolov8n.hef --input usb --labels-json /home/junyan/hailo/hailo-rpi5-examples/resources/yolov8n-labels.json
