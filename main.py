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

    clf.getSVC()
    # 3. slide_window to find the cars
    # 3.1 half of
    #print(img.shape)
    windows= detect.slide_window(img, x_start_stop=[0, img.shape[1]], y_start_stop=[img.shape[0]/2, img.shape[0]],
                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    hot_windows = detect.search_windows(image, windows, df.svc, df.XScaler, color_space=df.color_space,
                        spatial_size=df.spatial_size, hist_bins=df.hist_bins,
                        orient=df.orient, pix_per_cell=df.pix_per_cell,
                        cell_per_block=df.cell_per_block,
                        hog_channel=df.hog_channel, spatial_feat=df.spatial_feat,
                        hist_feat=df.hist_feat, hog_feat=df.hog_feat)
    draw_img = draw_boxes(img, hot_windows, color=(0, 0, 255), thick=6)
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
