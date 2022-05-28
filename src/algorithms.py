import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import math
from scipy.spatial.distance import cdist


class Algorithms:
    def get_all_contours(self, image=None):
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
            cnts, _ = cv.findContours(des, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
            return cnts
        else:
            contours = []
            DIR = "../dataset/cut"
            for f in os.listdir(DIR):
                f = os.path.join(DIR, f)
                image = cv.imread(f)
                # Converting to grayscale and then to binary image
                raw_gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                th, bin_img = cv.threshold(raw_gray_img, 1, 255, cv.THRESH_OTSU)
                des = cv.bitwise_not(bin_img)
                _, cnts, _ = cv.findContours(des, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
                # contour = np.array(contour)
                contours.append(cnts)
            return contours

    def get_all_poly_points(self, image=None, epsilon=0.01) -> np.ndarray:
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
                epsilon = epsilon * cv.arcLength(cnt, True)
                approx = cv.approxPolyDP(cnt, epsilon, True)
            return np.asarray(approx)
        else:
            all_polys = []
            contours = self.get_all_contours()
            for cnt in contours:
                cnt = np.array(cnt)
                epsilon = epsilon * cv.arcLength(cnt, True)
                approx = cv.approxPolyDP(cnt, epsilon, True)
                all_polys.append(approx)
            return all_polys

    def _calculate_compass_bearing(self, pt1, pt2):
        """
        Returns the bearing between two points.

        pt1
            A list representing the coordinates for the first point
        pt2
            A list representing the coordinates for the second point

        Returns:
          The bearing in degrees
        """
        lat1 = math.radians(pt1[0])
        lat2 = math.radians(pt2[0])

        diffLong = math.radians(pt2[1] - pt1[1])

        x = math.sin(diffLong) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (
            math.sin(lat1) * math.cos(lat2) * math.cos(diffLong)
        )

        initial_bearing = math.atan2(x, y)
        initial_bearing = math.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360

        return compass_bearing

    def _check_direction(self, degree):
        """
        Returns the direction based on the degree
        """
        if 0 <= degree < 45:
            return 0
        if 45 <= degree < 90:
            return 1
        if 90 <= degree < 135:
            return 2
        if 135 <= degree < 180:
            return 3
        if 180 <= degree < 225:
            return 4
        if 225 <= degree < 270:
            return 5
        if 270 <= degree < 315:
            return 6
        if 315 <= degree < 360:
            return 7

    def get_chain_code_histogram(self, image):
        histogram = [0] * 8
        contours = self.get_all_contours(image)
        contours = np.array(contours)
        contours = contours.reshape(contours.shape[1], 2)

        for idx in range(len(contours) - 1):
            degree = self._calculate_compass_bearing(contours[idx], contours[idx + 1])
            histogram[self._check_direction(degree)] += 1

        return histogram

    def _find_angle(self, M1, M2):
        """Returns the angle between two slopes"""

        angle = abs((M2 - M1) / (1 + M1 * M2))
        ret = math.atan(angle)

        # radian to degree
        val = math.degrees(ret)

        # Print the result
        return round(val)

    def _calculate_slope(self, pt1, pt2):
        """Returns the slope between two points"""
        if (pt1[0] - pt2[0]) == 0:
            return 1
        return (pt1[1] - pt2[1]) / (pt1[0] - pt2[0])

    def calculate_histogram(self, img):
        histogram = np.zeros([141, 180], dtype=np.uint8)
        approx = self.get_all_poly_points(img)
        approx = approx.reshape(approx.shape[0], approx.shape[2])
        lines = []
        for i in range(len(approx) - 1):
            lines.append([approx[i], approx[i + 1]])

        i = 0
        while i < len(lines):
            j = i + 1
            while j < len(lines):
                slope1 = self._calculate_slope(lines[i][0], lines[i][1])
                slope2 = self._calculate_slope(lines[j][0], lines[j][1])
                angle = self._find_angle(slope1, slope2)

                dists = cdist(lines[i], lines[j])
                max_value = round(dists.max())
                min_value = round(dists.min())
                histogram[max_value][angle] += 1
                histogram[min_value][angle] += 1
                j = j + 1
            i = i + 1

        col = np.sum(histogram, axis=0)
        row = np.sum(histogram, axis=1)

        return np.concatenate((row, col))
