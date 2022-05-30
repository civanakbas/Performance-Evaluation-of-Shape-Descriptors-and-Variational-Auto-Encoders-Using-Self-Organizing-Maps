from tokenize import String
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from algorithms import Algorithms
import os

# ----------UNCOMMEND THE EXERCISE YOU WISH-----------
Alg = Algorithms()

# ****** Getting all contours and polys at once ******
# Alg = Algorithms()
# contours = Alg.get_all_contours()
# print("Type of contours =             ",type(contours),"                  Length of contours =             ", len(contours), " num of calculated images ")
# print("Type of contours[0] =          ",type(contours[0]), "                  Length of contours[0] =          ", len(contours[0]), "   how many objects in that image")
# print("Type of contours[0][0] =       ",type(contours[0][0]), "         Length of contours[0][0] =       ", len(contours[0][0]),"  how many contour points ")
# print("Type of contours[0][0][0] =    ",type(contours[0][0][0]), "         Length of contours[0][0][0] =    ", len(contours[0][0][0]), "   how many elements in one contour point")
# print("Type of contours[0][0][0][0] = ",type(contours[0][0][0][0]), "         Length of contours[0][0][0][0] = ", len(contours[0][0][0][0]),"   coordination points")


# polys = Alg.get_all_poly_points()
# print("Type of polys = ",type(polys), "   Length of polys = ", len(polys))


# ******** Example of polygonomial approximation.********
# img = cv.imread("../dataset/cut/obj_21.png")
# Alg = Algorithms()
# contours = Alg.get_all_contours(image=img)

# img_1 = np.zeros([100, 100, 1], dtype=np.uint8)
# img_2 = np.zeros([100, 100, 1], dtype=np.uint8)
# img_3 = np.zeros([100, 100, 1], dtype=np.uint8)
# img_4 = np.zeros([100, 100, 1], dtype=np.uint8)
# img_1.fill(255)
# img_2.fill(255)
# img_3.fill(255)
# img_4.fill(255)

# for cnt in contours:
#     approx = Alg.get_all_poly_points(img)
#     print(type(approx), approx.shape)
#     cv.drawContours(img_1, [approx], 0, (0), 1)
#     hull = cv.convexHull(cnt)
#     cv.drawContours(img_2, [hull], 0, (0), 1)

#     approx = Alg.get_all_poly_points(img, 0.02)
#     cv.drawContours(img_3, [approx], 0, (0), 1)

# fig, axs = plt.subplots(nrows=2, ncols=2)
# axs[0, 0].set_title("Poly Aprx. | Epsilon = 0.01")
# axs[0, 1].set_title("Poly Aprx. | Epsilon = 0.02")
# axs[1, 0].set_title("Convex Hull")
# axs[1, 1].set_title("Original -> Not Contour")
# axs[0, 0].imshow(img_1)
# axs[1, 0].imshow(img_2)
# axs[0, 1].imshow(img_3)
# axs[1, 1].imshow(img)

# plt.show()


# ********Adjusting all data to the same format*******
# i = 999
# for f in os.listdir("../dataset/cut"):
#     f = os.path.join("../dataset/cut",f)

#     img = cv.imread(f)
#     raw_gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     th, bin_img = cv.threshold(raw_gray_img, 1, 255, cv.THRESH_OTSU)
#     des = cv.bitwise_not(bin_img)
#     _,contour,_ = cv.findContours(des,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
#     for cnt in contour:
#         cv.drawContours(des,[cnt],0,255,-1)

#     des = cv.bitwise_not(des)
#     cv.imwrite("../dataset/cut/obj_" + str(i) + ".png", des)
#     i -= 1


# ******** Example of chain code histogram for input image *******

# img = cv.imread("../dataset/cut/obj_2.png")
# histogram = Alg.get_chain_code_histogram(img)
# print(histogram)


# ******** Example of PGH for input image *******
# img = cv.imread("../dataset/cut/obj_2.png")
# print(Alg.calculate_histogram(img))

# ******** Example of ChordArc for input image ******** 
img = cv.imread("../dataset/cut/obj_930.png")
img_1 = np.zeros([100, 100, 1], dtype=np.uint8)
img_1.fill(255)
poly = Alg.get_all_poly_points(img)
print("Feature Set: ",Alg.get_chord_arc(poly))
cv.drawContours(img_1,[poly],-1,0,1)
plt.imshow(img_1)
plt.show()



#****** Example of get_vector_angle* ******
# points = np.array([[0,0],[0,2],[2,2],[0,2 + 2*np.sqrt(3)],[2,2 + 2*np.sqrt(3)],[4 + (2/np.sqrt(3)),0]])
# result = Alg.get_vector_angle(points)





#******** Example of Basic Methods **********

# img = cv.imread("../dataset/cut/obj_3.png")
# contours = Alg.get_all_contours(image=img)
# Alg.get_basics(img)


 
