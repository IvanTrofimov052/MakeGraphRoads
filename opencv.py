import cv2
import numpy as np
from matplotlib import pyplot as plt


# this class work with images
class OpenCv:
    def save_img(self, image, path):
        cv2.imwrite(path, image)

    def change_to_hsv(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        return hsv

    def make_mask(self, image, lower, upper):
        hsv = self.change_to_hsv(image)

        # define range of blue color in HSV
        lower_blue = np.array(lower)
        upper_blue = np.array(upper)

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []

        if (len(contours) >= 1):
            c = sorted(contours, key=cv2.contourArea, reverse=True)

            for cout in c:
                # compute the rotated bounding box of the largest contour
                rect = cv2.minAreaRect(cout)
                box = np.int0(cv2.boxPoints(rect))

                boxes.append(box)

                cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

        return image, boxes


class OpenCvWrapper(OpenCv):
    def detect_crosroad(self, img):
        image = np.array(img)[:, :, ::-1].copy()
        image, boxes = self.make_mask(image, [170, 120, 70], [180, 255, 255])

        areas = []

        for box in boxes:

            if not (box is None):
                a = ((box[0][0]-box[1][0])**2 + (box[0][1]-box[1][1])**2)**(1/2)
                b = ((box[2][0] - box[3][0]) ** 2 + (box[2][1] - box[3][1]) ** 2) ** (1 / 2)

                n = len(box)  # of corners

                area = 0.0
                for i in range(n):
                    j = (i + 1) % n
                    area += box[i][0] * box[j][1]
                    area -= box[j][0] * box[i][1]
                area = abs(area) / 2.0

                areas.append(area)
        try:
            return max(areas) > 3000
        except:
            return False

        self.save_img(image, "screen_2.png")