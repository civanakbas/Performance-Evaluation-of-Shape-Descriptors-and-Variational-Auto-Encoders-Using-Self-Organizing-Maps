import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class Algorithms():
    
    def get_contour_points(self,image: np.ndarray) -> np.ndarray:
        """
        Returns contours of the given image.

        Parameters:

        image (np.ndarray): Single image to calculating the contours.

        Returns:

        np.ndarray: It is shaped as (x,y,z,t). x for amount of image sent usually 1, y for amount of contour points, z for how many elements in one contour point , t for coordinate dimension of the point, 
        e.g., 
        
        shape = (1, 3, 1, 2)

        [[[

            [62,19],
            [58,14],
            [60,28]

                    ]]] . 
        """
        #Converting to grayscale and then to binary image
        raw_gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        th, bin_img = cv.threshold(raw_gray_img, 127, 255, cv.THRESH_OTSU)
        des = cv.bitwise_not(bin_img)
        _,contour,_ = cv.findContours(des,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
        contour = np.array(contour)
        return contour

    def get_poly_points(self, image: np.ndarray,epsilon=0.01) -> np.ndarray:
        """
        Returns polynomial approximation points of the given image.

        Parameters:

        image (np.ndarray): Single image for calculating the poly points.
        epsilon (float): a coefficient used for calculating the poly points. The resulting shape gets sharp when the value of epsilon gets big. Best to use as 0.01.(Which is default)

        Returns:

        np.ndarray: It is shaped as (x,y,z). X for how many points for polynomial approximation, Y for how many coordinates for one point, Z for coordinate dimension
        e.g., 
        
        shape = (3,1,2)

        [[[

            [62,19],
            [58,14],
            [60,28]

                    ]]] . 
        """
        contours = self.get_contour_points(image)
        for cnt in contours:
            epsilon = epsilon*cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, epsilon, True)

        return np.array(approx)
            
