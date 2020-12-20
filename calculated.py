from graph import *
from opencv import *
from roading_coords import *


# this class dines opencv and graph
class Converter:
    lenght_road = LengthRoad()
    graph = Graph()
    opencv = OpenCvWrapper()

    def analayze_image(self, img):
        has_crossroad = self.opencv.detect_crosroad(img)

        print(has_crossroad)


# this the main claculated class
class Calculated:
    converter = Converter()
    roading_coords = RoadingRobot()