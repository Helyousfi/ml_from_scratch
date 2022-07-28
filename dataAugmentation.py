# Data augmentation from scratch

import random

# bbox : X_TL, Y_TL, X_BR, Y_BR 

class horizontalFlip:
    def __init__(self, p = 0.5):
        self.p = p
    
    def __cal__(self, img, bboxes):
        w, h, _ = img.shape
        img = img[:, ::-1, :]
        if random.random() < self.p:
            bboxes[0] = img.shape[1] - bboxes[0]
            bboxes[2] = img.shape[1] - bboxes[2]
            temp = bboxes[3].copy()
            bboxes[3] = bboxes[1]
            bboxes[1] = temp
            return img, bboxes
