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

        if (len(contours) >= 1):
            c = sorted(contours, key=cv2.contourArea, reverse=True)

            for cout in c:
                # compute the rotated bounding box of the largest contour
                rect = cv2.minAreaRect(cout)
                box = np.int0(cv2.boxPoints(rect))

                cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

        return image


class OpenCvWrapper(OpenCv):
    def detect_crosroad(self, img):
        img = self.make_mask(img, [0,0,0], [100, 100, 100])

        self.save_img(img, "screen.png")