from ultralytics import YOLO

# 加载模型
model = YOLO(r"D:\daima\python\yolo\yolo_code\stripe\code_package\runs\detect\train20\weights\stripe.pt")

# 导出为固定尺寸的ONNX
model.export(
    format='onnx',
    imgsz=(640, 640),  # 固定输入尺寸
    dynamic=False,      # 禁用动态维度
    opset=12           # 必须≥12
)