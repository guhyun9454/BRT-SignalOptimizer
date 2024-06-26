from ultralytics import YOLO
import cv2
import math 
import numpy as np
from functions import draw_polygon, is_inside_polygon, bbox_intersects_polygon, draw_traffic_light
from ROI import ROI
import time

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
red = (0,0,255) #bgr
green = (0,255,0)
green_light = True
last_change_time = time.time()

model = YOLO('yolo-Weights/yolov8n.pt')

source = "videos/view1-1.mp4"
cap = cv2.VideoCapture(source)

SAVE = False

if SAVE:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 코덱
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

road = ROI((350, 550),(1600, 1080),(1920, 680),(350,500))

traffic_actuated_region = ROI((1100, 850),(1600, 1080),(1920, 680),(1550, 650))



while True:
    success, img = cap.read()
    if not success:
        break

    current_time = time.time()
    if current_time - last_change_time >= 3:
        green_light = not green_light
        last_change_time = current_time



    results = model(img, stream=True,classes = [0,5],device = "mps",conf = 0.5) 


    draw_polygon(img, road, (0, 0, 50), 0.5)
    draw_polygon(img, traffic_actuated_region, (50, 50, 0), 0.5)

    for r in results: #한 번
        boxes = r.boxes

        for box in boxes:
            #info abt box
            x1, y1, x3, y3 = box.xyxy[0] #x1,y1 : top left, x3,y3 :bottom right
            x1, y1, x3, y3 = int(x1), int(y1), int(x3), int(y3) # convert to int values
            x2,y2 = x1, y3

            cls = int(box.cls[0])
            class_name = classNames[cls]
            confidence = math.ceil((box.conf[0]*100))/100
            
            if class_name == "person":
                if is_inside_polygon((x2,y2),road) or is_inside_polygon((x3,y3),road):
                    #도로에서 사람이 있는 경우
                    cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x3, y3), color= red, thickness=3)
                else: #안전
                    cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x3, y3), color= green, thickness=3)

            elif class_name == "bus":
                if (bbox_intersects_polygon((x1,y1,x3,y3),traffic_actuated_region)):
                    #버스가 감응구역에 있는 경우
                    cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x3, y3), color=(0, 0, 255), thickness=3)
                    green_light = True
                else:
                    cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x3, y3), color=(255, 255, 255), thickness=3)


            cv2.putText(img,class_name , [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    draw_traffic_light(img, green_light)

    if SAVE:
        out.write(img)
    cv2.imshow('obj detection demo', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
if SAVE:
    out.release()   

cv2.destroyAllWindows()