import random
import cv2
import numpy as np

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
    def __init__(self, max_angle = 45):
        self.angle = random.uniform(0, max_angle)
        # Create rotation matrix
        self.rotation_matrix = np.array([    [np.cos(angle), -np.sin(angle)],
                                             [np.sin(angle), np.cos(angle) ]     ])
    
    def rotate(self, img, angle, pivot_point=[0,0]):
        rotated_img = np.zeros(img.shape)

        # Pivot point 
        pivot_point_x = pivot_point[0]
        pivot_point_y = pivot_point[1]

        # Get image dimensions
        h, w = img.shape

        # Get xp and yp using rotation matrix
        for y in range(h):
            for x in range(w):
                new_coords = np.matmul(self.rotation_matrix, np.array([y - pivot_point_y, x - pivot_point_x]).T)  
                xp = pivot_point_x + int(new_coords[1])
                yp = pivot_point_y + int(new_coords[0])

                # Test if we're still inside the image
                if xp < w and yp < h:
                    rotated_img[yp, xp] = img[y, x]

        return rotated_img
    
    def rotate_box(self, angle, bboxes):
        for k in range(len(bboxes)):
            [ [bboxes[k][0]] , [bboxes[k][1]] ] = np.matmul(self.rotation_matrix, [ [bboxes[k][0]] , [bboxes[k][1]] ] )
            [ [bboxes[k][2]] , [bboxes[k][3]] ] = np.matmul(self.rotation_matrix, [ [bboxes[k][2]] , [bboxes[k][3]] ] )

            tmp_x = bboxes[k][0]
            tmp_y = bboxes[k][1]

            bboxes[k][0] = min(bboxes[k][0], bboxes[k][2])
            bboxes[k][1] = min(bboxes[k][1], bboxes[k][3])

            bboxes[k][2] = max(tmp_x, bboxes[k][2])
            bboxes[k][3] = max(tmp_y, bboxes[k][3])

        return bboxes


    def __call__(self, img, bboxes):
        rotated_img = self.rotate(img, self.angle, [img.shape[1]/2, img.shape[0]/2])
        rotated_bboxes = self.rotate(self.angle, bboxes)
        return rotated_img, rotated_bboxes
