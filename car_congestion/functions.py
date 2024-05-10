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