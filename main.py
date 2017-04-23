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
svc = clf.getSVC()


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
