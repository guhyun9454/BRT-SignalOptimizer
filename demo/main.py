from ultralytics import YOLO
import cv2
import math 
import numpy as np
from functions import draw_polygon, is_inside_polygon
from ROI import ROI

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

model = YOLO('yolo-Weights/yolov8m.pt')

source = "videos/view2.mp4"
cap = cv2.VideoCapture(source)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 코덱
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

crosswalk = ROI((920, 450),(870, 580),(1280, 580),(1280, 450))

crosswalk2 = ROI((200, 420),(100, 470),(920, 450),(820, 400))



while True:
    success, img = cap.read()
    cmask = img.copy

    results = model(img, stream=True,classes = [0,5],device = "mps",conf = 0.5) 


    draw_polygon(img, crosswalk, (0, 0, 50), 0.5)
    draw_polygon(img, crosswalk2, (50, 50, 0), 0.5)

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
                if is_inside_polygon((x2,y2),crosswalk) or is_inside_polygon((x3,y3),crosswalk):
                    #crosswalk 1에서 건너는 경우 
                    cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x3, y3), color= red, thickness=3)
                elif is_inside_polygon((x2,y2),crosswalk2) or is_inside_polygon((x3,y3),crosswalk2):
                    #crosswalk 2에서 건너는 경우
                    cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x3, y3), color= red, thickness=3)
                else: #안전
                    cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x3, y3), color= green, thickness=3)

            elif class_name == "bus":
                cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x3, y3), color=(255, 255, 255), thickness=3)
                
            cv2.putText(img,class_name , [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # out.write(img)
    cv2.imshow('obj detection demo', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()

cv2.destroyAllWindows()