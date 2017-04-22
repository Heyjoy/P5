import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import *
from datafield import *
import classifier as clf
import extraction
def pipeline(image):

    # 3. slide_window to find the cars
    return image


# 1. decide what features to use
# 2. train the classifier
clf.datasetInit()
rand_car = np.random.choice(len(cars))
rand_notcar = np.random.choice(len(notcars))
this_car = mpimg.imread(cars[rand_car])
this_notcar = mpimg.imread(notcars[rand_notcar])

features = extraction.extract_features([cars[rand_car]],color_space='YCrCb',hog_channel='ALL',plot =True)
#twoImagePlot(this_car,this_notcar,title1="car",title2 ="no car",path="output_images/car_not_car.png")
