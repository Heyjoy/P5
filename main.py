import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import *
from datafield import *
import classifier as clf
import extraction
import detect
def pipeline(img):

    clf.dataInit()
    # 3. slide_window to find the cars
    # 3.1 half of
    #print(img.shape)
    ystart = 400
    ystop = 656
    scale = 1.5
    detect.find_cars(img, ystart, ystop, scale)
    return draw_img

# 1. decide what features to use
# 2. train the classifier

image = mpimg.imread('test_images/test1.jpg')
resImage = pipeline(image)
#twoImagePlot(image,resImage)
plt.imshow(resImage)
plt.show()



'''rand_car = np.random.choice(len(cars))
rand_notcar = np.random.choice(len(notcars))
this_car = mpimg.imread(cars[rand_car])
this_notcar = mpimg.imread(notcars[rand_notcar])
print(rand_car,rand_notcar)
rand_car = 1000
rand_notcar = 2000
features = extraction.extract_features([cars[rand_car]],color_space, spatial_size,
                        hist_bins, orient,
                        pix_per_cell, cell_per_block, hog_channel,
                        spatial_feat, hist_feat, hog_feat,plot = True)'''
