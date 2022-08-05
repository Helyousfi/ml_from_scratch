import json
import cv2, os, sys
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class laodDataset:
    def __init__(self, json_path):
        self.json_file = json.load(open(json_path))

    def __call__(self, index):
        image_dict = self.json_file["images"][index]
        image_id = image_dict["id"]
        image = cv2.imread(os.path.join("dataset/train", image_dict["file_name"]), 0)
        #image = cv2.resize(image, dsize=(400, 400))
        
        bboxes = []
        categories = []
        bbox_dict = self.json_file["annotations"]
        for ann in bbox_dict:
            if ann["image_id"] == image_id:
                bboxes.append(ann["bbox"])
                categories.append(ann["category_id"])
        
        return image, bboxes, categories

def showImageBbox(image, bboxes):
    _, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    for bbox in bboxes:
        print((bbox[0], bbox[1]), bbox[2], bbox[3])
        p = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill = 0, linewidth=1)
        ax.add_patch(p)
    plt.show()


class RandomRotate:
    def __init__(self, max_angle = 45):
        self.angle = 3.14 * random.uniform(0, max_angle) / 180
        # Create rotation matrix
        self.rotation_matrix = np.array([    [np.cos(self.angle), -np.sin(self.angle)],
                                             [np.sin(self.angle), np.cos(self.angle) ]     ])
    
    def rotate(self, img, pivot_point=[0,0]):
        rotated_img = np.zeros(img.shape)

        # Pivot point 
        pivot_point_x = pivot_point[0]
        pivot_point_y = pivot_point[1]

        # Get image dimensions
        h, w = img.shape

        # Get xp and yp using rotation matrix
        for y in range(h):
            for x in range(w):
                new_coords = np.matmul(self.rotation_matrix, np.array([x - pivot_point_x, y - pivot_point_y]).T)  

                xp = int(pivot_point_x + new_coords[0])
                yp = int(pivot_point_y + new_coords[1])

                # Test if we're still inside the image
                if 0 < xp < w and 0 < yp < h:
                    rotated_img[yp, xp] = img[y, x]

        return rotated_img
    
    def rotate_box(self, bboxes, pivot_point=[0,0]):
        pivot_point_x = pivot_point[0]
        pivot_point_y = pivot_point[1]
        for k in range(len(bboxes)):
            [ x1 , y1 ] = np.matmul(self.rotation_matrix, np.array([bboxes[k][0]-pivot_point_x,  bboxes[k][1]-pivot_point_y]).T )
            [ x2 , y2 ] = np.matmul(self.rotation_matrix, np.array([bboxes[k][0]+bboxes[k][2]-pivot_point_x , bboxes[k][1]-pivot_point_y]).T )
            [ x3 , y3 ] = np.matmul(self.rotation_matrix, np.array([bboxes[k][0]-pivot_point_x , bboxes[k][1]+bboxes[k][3]-pivot_point_y]).T)
            [ x4 , y4 ] = np.matmul(self.rotation_matrix, np.array([bboxes[k][0]+bboxes[k][2]-pivot_point_x  , bboxes[k][1]+bboxes[k][3]-pivot_point_y]).T)

            bboxes[k][0] = min(pivot_point_x + x1, pivot_point_x + x3)
            bboxes[k][1] = min(pivot_point_y + y1, pivot_point_y + y2)

            bboxes[k][2] = max(pivot_point_x + x2, pivot_point_x + x4) - bboxes[k][0]
            bboxes[k][3] = max(pivot_point_y + y3, pivot_point_y + y4) - bboxes[k][1]

        return bboxes


    def __call__(self, img, bboxes):
        rotated_img = self.rotate(img, [img.shape[1]/2, img.shape[0]/2])
        rotated_bboxes = self.rotate_box(bboxes, [img.shape[1]/2, img.shape[0]/2])
        return rotated_img, rotated_bboxes


class RandomCrop:
    def __init__(self, height_r = 0.5, width_r = 0.5):
        self.width_r = width_r
        self.height_r = height_r 
    
    def __call__(self, img, bboxes):
        assert self.width_r <= 1, "height ratio must be lesser than 1"
        assert self.height_r <=1, "width ratio must be lesser than 1"

        width = self.width_r * img.shape[1]
        height = self.height_r * img.shape[0]  

        x1 = int(random.uniform(0, img.shape[1] - width))
        x2 = int(x1 + width)
        y1 = int(random.uniform(0, img.shape[0] - height))
        y2 = int(y1 + height)

        print(y1, y2, x1, x2)
        # Crop the image
        img = img[y1:y2, x1:x2]

        # Crop the bboxes
        for k in range(len(bboxes)):
            temp_x = bboxes[k][0]
            temp_y = bboxes[k][1]
            bboxes[k][0] = max(bboxes[k][0] - x1, 0)
            bboxes[k][1] = max(bboxes[k][1] - y1, 0)
            bboxes[k][2] = min(temp_x + bboxes[k][2], x2) - max(x1, temp_x)
            bboxes[k][3] = min(temp_y + bboxes[k][3], y2) - max(y1,temp_y)
        
        return img, bboxes


def swap(a, b):
    return b, a

# bbox : X_TL, Y_TL, X_BR, Y_BR 
class HorizontalFlip:
    def __init__(self, p = 0.5):
        self.p = p
    
    def __call__(self, img, bboxes):
        if random.random() < self.p:
            img = img[:, ::-1]
            _, width = img.shape
            for k in range(len(bboxes)):
                bboxes[k][0] = (width - bboxes[k][0]) - bboxes[k][2]

            return img, bboxes


if __name__ == "__main__":
    dataset = laodDataset("dataset/train/annotations.json")
    image, bboxes, _ = dataset(1)

    showImageBbox(image, bboxes)

    HorizontalFlip = HorizontalFlip()
    rotated_img, rotated_bboxes = HorizontalFlip(image, bboxes)

    showImageBbox(rotated_img, rotated_bboxes)

    plt.imshow(rotated_img, cmap="gray")
    plt.show()



    