from ultralytics import YOLO
import cv2
import numpy as np
from ROI import ROI
from functions import draw_polygon

# YOLO 모델 로드
model = YOLO('yolo-Weights/yolov8m.pt')

# 비디오 파일 경로
source = "videos/view3.mp4"

# 대상 클래스의 인덱스
class_indices = [2, 7, 5]

# 비디오 캡처
cap = cv2.VideoCapture(source)

# 교차보도 영역 좌표
crosswalk = ROI((540, 166), (209, 836), (1499, 834), (890, 171))

while True:
    success, img = cap.read()
    cmask = img.copy()

    # 객체 수 초기화
    current_count = 0

    # 객체 검출
    results = model(img, stream=True, classes=class_indices, device="mps", conf=0.5)

    # 객체 감지된 바운딩 박스 그리기 및 수 세기
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            
            # 바운딩 박스 중심 좌표
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # 교차보도 영역 내에 있는지 확인
            if crosswalk.is_inside((center_x, center_y)):
                current_count += 1

        # 교차보도 영역 그리기
    draw_polygon(img, crosswalk, (0, 0, 50), 0.5)

    # 객체 수 출력
    cv2.putText(img, f'Total Count: {current_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Object Detection Demo', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
