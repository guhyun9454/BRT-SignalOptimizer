from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Define path to video file
source = "videos/view1.mp4"

# Run inference on the source
results = model(source,show = True,conf = 0.5,device = "mps")  # generator of Results objects

