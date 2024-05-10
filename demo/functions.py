import cv2
import numpy as np


def draw_polygon(img, ROI, color, alpha = 0.5):
    """
    img(cv2.img):
    폴리곤을 그릴 arr

    ROI (ROI):
    ROI객체
    """
    overlay = img.copy()
    cv2.fillPoly(overlay, [ROI.polygon], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def is_inside_polygon(point, ROI):
    "ROI 안에 point의 존재 여부를 리턴합니다."
    return cv2.pointPolygonTest(contour=ROI.polygon, pt=point, measureDist=False) >= 0


def bbox_intersects_polygon(bbox, roi):
    """
    ROI와 bbox가 겹치는 여부를 리턴합니다.

    - bbox: Tuple of (x1, y1, x3, y3) where (x1, y1) is the top-left and (x3, y3) is the bottom-right of the bbox.
    - roi: ROI object containing a polygon.
    """
    x1, y1, x3, y3 = bbox
    bbox_contour = np.array([(x1, y1), (x3, y1), (x3, y3), (x1, y3)])
    bbox_contour = bbox_contour.reshape((-1, 1, 2)).astype(np.int32)

    intersection = cv2.intersectConvexConvex(bbox_contour, roi.polygon)
    return intersection[0] > 0

def draw_traffic_light(img, green_light):
    radius = 60
    thickness = -1  
    light_x = img.shape[1] - 180  
    red = (0, 0, 255)
    green = (0, 255, 0)
    gray = (100, 100, 100)

    # 빨간불
    red_color = red if not green_light else gray
    cv2.circle(img, (light_x, 90), radius, red_color, thickness)

    # 초록불
    green_color = green if green_light else gray
    cv2.circle(img, (light_x, 210), radius, green_color, thickness)