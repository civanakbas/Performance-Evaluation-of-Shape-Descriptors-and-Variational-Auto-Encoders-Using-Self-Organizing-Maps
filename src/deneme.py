from tokenize import String
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from algorithms import Algorithms

img = cv.imread("../dataset/cut/obj_68.png")
Alg = Algorithms()
contours = Alg.get_contour_points(image=img)


img_1 = np.zeros([100,100,1],dtype=np.uint8)
img_2 = np.zeros([100,100,1],dtype=np.uint8)
img_3 = np.zeros([100,100,1],dtype=np.uint8)
img_4 = np.zeros([100,100,1],dtype=np.uint8)
img_1.fill(255)
img_2.fill(255)
img_3.fill(255)
img_4.fill(255)

for cnt in contours:
    approx = Alg.get_poly_points(img)
    print(type(approx), approx.shape)
    cv.drawContours(img_1, [approx], 0, (0), 1)
    hull = cv.convexHull(cnt)
    cv.drawContours(img_2, [hull], 0, (0), 1)

    approx = Alg.get_poly_points(img,0.02)
    cv.drawContours(img_3, [approx], 0, (0), 1)

fig, axs = plt.subplots(nrows=2,ncols=2)
axs[0,0].set_title("Poly Aprx. | Epsilon = 0.01")
axs[0,1].set_title("Poly Aprx. | Epsilon = 0.02")
axs[1,0].set_title("Convex Hull")
axs[1,1].set_title("Original -> Not Contour")
axs[0,0].imshow(img_1)
axs[1,0].imshow(img_2)
axs[0,1].imshow(img_3)
axs[1,1].imshow(img)

plt.show()
