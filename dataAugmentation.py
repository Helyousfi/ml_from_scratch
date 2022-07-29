import random
import cv2

def swap(a, b):
    return b, a

# bbox : X_TL, Y_TL, X_BR, Y_BR 
class HorizontalFlip:
    def __init__(self, p = 0.5):
        self.p = p
    
    def __cal__(self, img, bboxes):
        if random.random() < self.p:
            img = img[:, ::-1, :]

            for k in range(len(bboxes)):
                # change x's
                bboxes[k][1] = img.shape[0] - bboxes[k][1]
                bboxes[k][3] = img.shape[0] - bboxes[k][3]
                # change y's
                bboxes[k][0], bboxes[k][2] = swap(bboxes[k][0], bboxes[k][2])
            return img, bboxes

class VerticalFlip:
    def __init__(self, p = 0.5):
        self.p = p 

    def __cal__(self, img, bboxes):
        if random.random() < self.p:
            img = img[::-1, :, :]

            for k in range(len(bboxes)):
                bboxes[k][1] = img.shape[0] - bboxes[k][1]
                bboxes[k][3] = img.shape[0] - bboxes[k][3]

                bboxes[k][0], bboxes[k][2] = swap(bboxes[k][0], bboxes[k][2])

            return img, bboxes

class RandomScale:
    def __init__(self, scale = 0.5):
        """
        Randomly scales the image from (1 - scale) to (1 + scale).
        bboxes = [[x_min, y_min, x_max, y_max], [x_min, y_min, x_max, y_max]]
        """
        self.scale = scale

    def __cal__(self, img, bboxes):
        assert self.scale > 0, "scale must be a positive number !"
        assert self.scale < 1, "scale must be lesser than 1 !"

        scale_x = random.uniform(1 - self.scale, 1 + self.scale)
        scale_y = random.uniform(1 - self.scale, 1 + self.scale)

        dsize = (img.shape[1] * scale_x, img.shape[0] * scale_y)
        img = cv2.resize(img, dsize)
        for k in range(len(bboxes)):
            for i in range(3):
                if i%2 == 0:
                    bboxes[k][i] =  bboxes[k][i] * scale_x
                else:
                    bboxes[k][i] =  bboxes[k][i] * scale_y

        return img, bboxes

class RandomCrop:
    def __init__(self, height_r = 0.5, width_r = 0.5):
        self.width_r = width
        self.height_r = height 
    
    def __cal__(self, img, bboxes):
        
        assert height_r <= 1, "height ratio must be lesser than 1"
        assert width_r <=1, "width ratio must be lesser than 1"

        width = self.width_r * img.shape[1]
        height = self.height_r * img.shape[0]  

        x1 = random.uniform(0, img.shape[1] - width)
        x2 = x1 + width
        y1 = random.uniform(0, img.shape[0] - height)
        y2 = y1 + height

        # Crop the image
        img = img[y1:y2,x1:x2,:]

        # Crop the bboxes
        for k in range(len(bboxes)):
            bboxes[k][0] = max(bboxes[k][0], x1)
            bboxes[k][1] = max(bboxes[k][1], y1)
            bboxes[k][2] = max(bboxes[k][2], x2)
            bboxes[k][3] = max(bboxes[k][3], y2)
        
        return img, bboxes

class RandomRotate:
    def __init__(self, angle = 45):
        self.angle = angle
    
    def __call__(self, img, bboxes):
        



