import cv2 as cv
import os

#20 ve 55. obje kontrol et çift kenarlı çıkıyor
directory = "dataset/cut"
fileNameCounter = 1

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    raw_img = cv.imread(f)
    
    #Converting to grayscale and then to binary image
    raw_gray_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)
    th, bin_img = cv.threshold(raw_gray_img, 127, 255, cv.THRESH_OTSU)
    des = cv.bitwise_not(bin_img)
    _,contour,_ = cv.findContours(des,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        cv.drawContours(des,[cnt],0,255,-1)
    
    final = cv.bitwise_not(des)
    
    #Canny edge detection and finding contours
    edged = cv.Canny(final, 30, 200)

    cv.imwrite("dataset/contours/obj_"+str(fileNameCounter)+".png",edged)
    fileNameCounter += 1
    

