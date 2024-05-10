import numpy as np
import cv2 

class ROI:
    def __init__(self,p1,p2,p3,p4):
        (x1,y1),(x2,y2),(x3,y3),(x4,y4) = p1,p2,p3,p4
        self.polygon = np.array([p1, p2, p3, p4], np.int32).reshape((-1, 1, 2))
    def is_inside(self, point):
        return cv2.pointPolygonTest(self.polygon, point, False) >= 0