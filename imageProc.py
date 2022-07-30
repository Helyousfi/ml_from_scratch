import numpy as np
import matplotlib.pyplot as plt
import cv2

def convolution(image, convFiler):
    image_out = np.zeros(image.shape)
    for y in range(1, image.shape[0]-1):
        for x in range(1, image.shape[1]-1):
            S = 0
            for n in range(-convFiler.shape[1]//2, convFiler.shape[1]//2 + 1):
                for m in range(-convFiler.shape[0]//2, convFiler.shape[1]//2 + 1):
                    S += image[y + n, x + m] * convFiler[n + convFiler.shape[1]//2, m + convFiler.shape[0]//2]
            image_out[y, x] = S
    return image_out

convFiler = np.array(  [[-1, -1, -1],
                        [-1,  9, -1],
                        [-1, -1, -1]]  )

image = cv2.imread("dataset/train/car01_jpg.rf.4825426598b0deaf0dae031c7f15797c.jpg", 0)
image = cv2.resize(image, dsize=(200,200))

image_out = convolution(image, convFiler)

plt.imshow(image_out)
plt.show()