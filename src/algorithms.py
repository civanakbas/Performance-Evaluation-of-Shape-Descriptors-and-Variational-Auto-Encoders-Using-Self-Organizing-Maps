from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import os
import math
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev


class Algorithms:
    def __init__(self) -> None:
        self.sinc_lut = {}
        for i in range(360):
            res = round(math.degrees(np.sinc(math.radians(i) / np.pi)), 4)
            self.sinc_lut[res] = i
            
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
                des = cv.bitwise_not(image)
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
    def _smoothen_image(self,contours):

        smoothened = []
        for contour in contours:
            x,y = contour.T
            x = x.tolist()[0]
            y = y.tolist()[0]
            tck, u = splprep([x,y], u=None, s=1.0, per=1)
            u_new = np.linspace(u.min(), u.max(), 25)
            x_new, y_new = splev(u_new, tck, der=0)
            res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
            smoothened.append(np.asarray(res_array, dtype=np.int32))
        return smoothened

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
        contours = self._smoothen_image(contours)
        contours = np.array(contours)
        contours = contours.reshape(contours.shape[1], 2)

        for idx in range(len(contours) - 1):
            degree = self._calculate_compass_bearing(contours[idx], contours[idx + 1])
            histogram[self._check_direction(degree)] += 1
        n_histogram = [item / len(contours) for item in histogram]

        return n_histogram

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

        # Calculate feature vector f_pgh using conditional probabilites
        f_pgh_row = []
        for idx, row in enumerate(histogram):
            row_sum_j = 0
            row_sum = 0
            for val in row:
                row_sum_j += (idx+1)*val
                row_sum += val
            # print(f"At row {idx}, the row_sum_j is {row_sum_j}, the row_sum is {row_sum}")
            if row_sum == 0:
                f_pgh_row.append(0)
            else:
                f_pgh_row.append(row_sum_j / row_sum)
        f_pgh_col = []

        for idx, col in enumerate(histogram.T):
            col_sum_j = 0
            col_sum = 0
            for val in col:
                col_sum_j += (idx+1)*val
                col_sum += val
            # print(f"At row {idx}, the row_sum_j is {col_sum_j}, the row_sum is {col_sum}")
            if col_sum == 0:
                f_pgh_col.append(0)
            else:
                f_pgh_col.append(col_sum_j / col_sum)

        return np.concatenate((f_pgh_row,f_pgh_col))

    def _cross_product(self, A):

        x1 = A[1, 0, 0] - A[0, 0, 0]
        y1 = A[1, 0, 1] - A[0, 0, 1]

        x2 = A[2, 0, 0] - A[0, 0, 0]
        y2 = A[2, 0, 1] - A[0, 0, 1]

        return x1 * y2 - y1 * x2

    def _get_sinc_degree(self, lut: dict, x):
        keys = lut.keys()
        min = np.inf
        for key in keys:
            dist = np.abs(x - key)
            if dist < min:
                min = dist
                res = key
        return res

    def get_chord_arc(self, points):
        """Returns chordArc value of given polygon

        Parameters:

        points (ndarray): Contour points from polynomial approximation

        Returns:
        
        A float feature set = [Sum of convex angles, Sum of concave angles, Mean of convex angles, Mean of concave angles, Variance of convex angles, Variance of concave angles, Total number of angles]

        """
        start = points[0]
        stop = points[2]
        arc_length = np.linalg.norm(points[0] - points[1])
        N = len(points)
        curr = 0
        prev = 0
        convex_angles = []
        concave_angles = []
        if cv.isContourConvex(points):
            convex_angles.append(self.sinc_lut[0])
        else:
            for i in range(N):

                temp = [points[i], points[(i + 1) % N], points[(i + 2) % N]]
                temp = np.array(temp)
                curr = self._cross_product(temp)

                if curr < 0:  # convex
                    if prev > 0:
                        stop = temp[1]
                        x = round(
                            math.degrees(np.linalg.norm(start - stop) / arc_length)
                        ,4)
                        x = self._get_sinc_degree(self.sinc_lut, x)
                        concave_angles.append(self.sinc_lut[x])
                        start = temp[0]
                        arc_length = np.linalg.norm(temp[0] - temp[1]) + np.linalg.norm(
                            temp[1] - temp[2]
                        )
                        #print("concave bulduk ve ekledik")
                        #print("konveks")

                    else:
                        arc_length += np.linalg.norm(temp[1] - temp[2])
                        #print("konveks")

                    if (
                        temp[2, 0, 0] == points[0, 0, 0]
                        and temp[2, 0, 1] == points[0, 0, 1]
                    ):
                        stop = temp[2]
                        x = round(
                            math.degrees(np.linalg.norm(start - stop) / arc_length)
                        ,4)
                        x = self._get_sinc_degree(self.sinc_lut, x)
                        if curr > 0:
                            concave_angles.append(self.sinc_lut[x])
                            #print("concave bulduk ve ekledik")
                        elif curr < 0:
                            convex_angles.append(self.sinc_lut[x])
                            #print("convex bulduk ve ekledik")
                        break

                else:
                    if prev < 0:
                        stop = temp[1]
                        x = round(
                            math.degrees(np.linalg.norm(start - stop) / arc_length)
                        ,4)
                        x = self._get_sinc_degree(self.sinc_lut, x)
                        #print("convex bulduk ve ekledik")
                        convex_angles.append(self.sinc_lut[x])
                        start = temp[0]
                        arc_length = np.linalg.norm(temp[0] - temp[1]) + np.linalg.norm(
                            temp[1] - temp[2]
                        )
                        #print("konkav")
                    else:
                        arc_length += np.linalg.norm(temp[1] - temp[2])
                        #print("konkav")

                    if (
                        temp[2, 0, 0] == points[0, 0, 0]
                        and temp[2, 0, 1] == points[0, 0, 1]
                    ):
                        stop = temp[2]
                        x = round(
                            math.degrees(np.linalg.norm(start - stop) / arc_length)
                        ,4)
                        x = self._get_sinc_degree(self.sinc_lut, x)
                        if curr > 0:
                            concave_angles.append(self.sinc_lut[x])
                            #print("concave bulduk ve ekledik")
                        elif curr < 0:
                            convex_angles.append(self.sinc_lut[x])
                            #print("convex bulduk ve ekledik")
                        break

                prev = curr

        sumvex = np.sum(convex_angles)
        sumcave = np.sum(concave_angles)
        if len(convex_angles) == 0:
            mean_vex = 0
            var_vex = 0
        else:
            mean_vex = sumvex / len(convex_angles)
            var_vex = np.var(convex_angles)
        if len(concave_angles) == 0:
            mean_cave = 0
            var_cave = 0
        else:
            mean_cave = sumcave / len(concave_angles)
            var_cave = np.var(concave_angles)

        total_arcs = len(convex_angles) + len(concave_angles)
        # print("--------------------")
        # print("Concave angles: ", concave_angles)
        # print("Convex angles: ", convex_angles)
        return np.array(
            [sumvex, sumcave, mean_vex, mean_cave, var_vex, var_cave, total_arcs]
        )

    def _get_vector_angle(self, points):
        """Returns float degree between two adjacent vector"""

        # edge_1 = np.linalg.norm(points[0]-points[1])
        # edge_2 = np.linalg.norm(points[1]-points[2])
        # edge_3 = np.linalg.norm(points[0]-points[2])

        angles = []
        N = len(points)
        for i in range(N):
            temp = [points[i], points[(i + 1) % N], points[(i + 2) % N]]
            edge_1 = np.linalg.norm(temp[0] - temp[1])
            edge_2 = np.linalg.norm(temp[1] - temp[2])
            edge_3 = np.linalg.norm(temp[0] - temp[2])
            cosx = (edge_3 ** 2 - edge_1 ** 2 - edge_2 ** 2) / (-2 * edge_1 * edge_2)
            x = math.degrees(np.arccos(cosx))
            angles.append(x)
        # cosx = (edge_3**2 - edge_1**2 - edge_2**2) / (-2 * edge_1 * edge_2)

        # x = math.degrees(np.arccos(cosx))
        return angles

    def get_basics(self, image):

        contours = self.get_all_contours(image)
        img_contour = np.zeros([100, 100, 1], dtype=np.uint8)
        #img_hull = np.zeros([100, 100, 1], dtype=np.uint8)
        img_contour.fill(255)
        #img_hull.fill(255)
        for cnt in contours:
            cv.drawContours(img_contour, [cnt], 0, (0), 1)      
            hull = cv.convexHull(cnt)
            #cv.drawContours(img_hull, [hull], 0, (0), 1)
        mu = cv.moments(contours[0])     
        mc = [
            mu["m10"] / (mu["m00"] ),
            mu["m01"] / (mu["m00"] ),
          ]  # 1e-5 for preventing zero divison

        # Calculating the convexity
        perimeter = round(cv.arcLength(contours[0], True))
        hull_perimeter = cv.arcLength(hull, True)
        #hull_perimeter = 0
        x = []
        y = []
        for i in range(len(img_contour)):
            for j in range(len(img_contour[0])):
                if img_contour[i, j, 0] == 0:
                    x.append(i)
                    y.append(j)
        # for i in range(len(img_hull)):
        #     for j in range(len(img_hull)):
        #         if img_hull[i, j, 0] == 0:
        #             hull_perimeter += 1
        
        
        method_1 = hull_perimeter / perimeter
        print("Convexity :", method_1)

        # Calculating the main axis
        vx = np.var(x)
        vy = np.var(y)
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        sum = 0
        for i in range(len(x)):
            sum += (x[i] - mean_x) * (y[i] - mean_y)
        cov = sum / (len(x) - 1)
        e1 = vy + vx - np.sqrt(((vx + vy) ** 2) - 4 * (vx * vy - (cov ** 2)))
        e2 = vy + vx + np.sqrt(((vx + vy) ** 2) - 4 * (vx * vy - (cov ** 2)))
        method_2 = e1 / e2
        print("Temel Eksenler :", method_2)

        # Compactness
        area = mu["m00"]
        # area = 0
        # for i in range(len(image)):
        #     for j in range(len(image[0])):
        #         if image[i,j,0] ==  0:
        #             area += 1
        method_3 =  (perimeter ** 2) / (4 * np.pi * area)
        print("Compactness: ", method_3)


        # Circular Variance
        dists_to_var = []
        radius = round(perimeter / (2 * math.pi))
        lenx = len(x)
        for i in range(lenx):
            dists_to_var.append(np.abs(np.linalg.norm(mc - np.array([x[i],y[i]])) - radius))
        #diff = np.abs(np.sqrt((mc[0]-real_cnt[0][0])**2 + (mc[1] - real_cnt[0][1])**2) - radius)
        mean_dist = np.sum(dists_to_var) / lenx
        summy = 0
        for dist in dists_to_var:
            summy += (mean_dist - dist) ** 2
        method_4 = summy / ((lenx - 1) * (radius ** 2))
        print("Circular Variance: ", method_4)
        return np.array([method_1, method_2, method_3, method_4])
