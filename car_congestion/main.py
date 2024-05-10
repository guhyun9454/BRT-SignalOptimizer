from ultralytics import YOLO
import cv2
import numpy as np
from ROI import ROI
from functions import draw_polygon

# YOLO model loading
model = YOLO('yolo-Weights/yolov8m.pt')

# Video file path
source = "videos/view3.mp4"

# Target class indices
class_indices = [2, 7, 5]

# Video capture
cap = cv2.VideoCapture(source)

# Crosswalk area coordinates
crosswalk = ROI((540, 166), (209, 836), (1499, 834), (890, 171))

while True:
    success, img = cap.read()
    cmask = img.copy()

    # Initializing the number of objects
    current_count = 0

    # Object detection
    results = model(img, stream=True, classes=class_indices, device="mps", conf=0.5)

    # Drawing detected bounding boxes and counting objects
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            
            # Calculating the center coordinates of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Checking if the bounding box is inside the crosswalk area
            if crosswalk.is_inside((center_x, center_y)):
                current_count += 1

    # Drawing the crosswalk area
    draw_polygon(img, crosswalk, (0, 0, 50), 0.5)

   # Assigning message and color based on the number of objects
    if current_count <= 1:
        message = "Sparse"
        color = (0, 255, 0)  # Green
    elif current_count <= 4:
        message = "Normal"
        color = (0, 255, 255)  # Yellow
    else:
        message = "Congested"
        color = (0, 0, 255)  # Red

    # Displaying the number of objects and congestion level with color
    cv2.putText(img, f'Total Count: {current_count}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,255,255), 5)
    cv2.putText(img, f'Congestion: {message}', (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 5)

    cv2.imshow('Object Detection Demo', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
