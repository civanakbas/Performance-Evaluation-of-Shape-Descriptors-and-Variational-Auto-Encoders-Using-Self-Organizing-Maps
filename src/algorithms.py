import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os 

class Algorithms():
    
    def get_all_contours(self,image=None):
        """
        Returns all of the contours from the dataset.

        Returns:

        list: It is shaped as (x,y,z,t). x for amount of image sent usually 1, y for amount of contour points, z for how many elements in one contour point , t for coordinate dimension of the point, 
        e.g., 
        
        shape = (1, 3, 1, 2)

        [[[

            [62,19],
            [58,14],
            [60,28]

                    ]]] . 
        """
        if image is not None:
            raw_gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            th, bin_img = cv.threshold(raw_gray_img, 1, 255, cv.THRESH_OTSU)
            des = cv.bitwise_not(bin_img)
            _,cnts,_ = cv.findContours(des,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
            return cnts
        else:
            contours = []
            DIR = "../dataset/cut"
            for f in os.listdir(DIR):
                f = os.path.join(DIR,f)
                image = cv.imread(f)
                #Converting to grayscale and then to binary image
                raw_gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                th, bin_img = cv.threshold(raw_gray_img, 1, 255, cv.THRESH_OTSU)
                des = cv.bitwise_not(bin_img)
                _,cnts,_ = cv.findContours(des,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
                #contour = np.array(contour)
                contours.append(cnts)
            return contours

    def get_all_poly_points(self,image=None,epsilon=0.01) -> np.ndarray:
        """
        Returns all polynomial approximation points from the dataset.

        Parameters:

        image (np.ndarray): Single image for calculating the poly points.
        epsilon (float): a coefficient used for calculating the poly points. The resulting shape gets sharp when the value of epsilon gets big. Best to use as 0.01.(Which is default)

        Returns:

        list: It is shaped as (x,y,z). X for how many points for polynomial approximation, Y for how many coordinates for one point, Z for coordinate dimension
        e.g., 
        
        shape = (3,1,2)

        [[[

            [62,19],
            [58,14],
            [60,28]

                    ]]] . 
        """
        if image is not None:
            contours = self.get_all_contours(image)
            for cnt in contours:     
                epsilon = epsilon*cv.arcLength(cnt, True)
                approx = cv.approxPolyDP(cnt, epsilon, True)
            return approx
        else:
            all_polys = []
            contours = self.get_all_contours()
            for cnt in contours:
                cnt = np.array(cnt)
                epsilon = epsilon*cv.arcLength(cnt, True)
                approx = cv.approxPolyDP(cnt, epsilon, True)
                all_polys.append(approx)
            return all_polys
            
