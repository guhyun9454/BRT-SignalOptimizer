import cv2



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