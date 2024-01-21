import os
import platform
from ultralytics import YOLO
from y_config import Y_cfg


model_path = Y_cfg['out_file']
model = YOLO(f'runs/detect/{model_path}/weights/best.pt')  # load a custom trained model

# Export the model
model.export(format='engine',half=True)
model.export(format='openvino')
model.export(format='onnx')