import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import *
from datafield import *
import classifier as clf
import extraction
import detect
from moviepy.editor import VideoFileClip
def pipeline(img):

    # 3. use hog to find the cars
    draw_img =np.copy(img)
    ystart= 400
    ystop= 656
    scale = 1.5
    cells_per_step=2
    hotwindows=[]
    box_list=detect.find_cars(img, ystart, ystop, scale,cells_per_step,
                    df.svc, df.XScaler, df.orient,df.pix_per_cell,df.cell_per_block,
                    df.spatial_size, df.hist_bins,df.spatial_feat,df.hist_feat,df.hog_feat)
    hotwindows.extend(box_list)

    ystart= 400
    ystop= 496
    scale = 1
    cells_per_step=2
    box_list=detect.find_cars(img, ystart, ystop, scale,cells_per_step,
                    df.svc, df.XScaler, df.orient,df.pix_per_cell,df.cell_per_block,
                    df.spatial_size, df.hist_bins,df.spatial_feat,df.hist_feat,df.hog_feat)
    hotwindows.extend(box_list)
    draw_img =detect.heatmap(img,hotwindows,10)
    #heatmap,draw_img =detect.heatmapImage(img,hotwindows,2)
    #raw_img = draw_boxes(img,hotwindows)
    return draw_img

# 1. decide what features to use
# 2. if classifier no trained, train the classifier
clf.dataInit()

# image = mpimg.imread('test_images/test1.jpg')
# resImage = pipeline(image)
# #twoImagePlot(image,resImage)
# plt.imshow(resImage)
# plt.show()

# plt.subplots(6, 3, figsize=(16, 28))
# for i in range(6):
#     img = plt.imread('test_images/test{}.jpg'.format(i+1), format='RGB')
#     raw_img,heatmap,draw_img = pipeline(img)
#     plt.subplot(6,3,i*3+1)
#     plt.imshow(raw_img)
#     if i == 0: plt.title('Raw with windows')
#
#     plt.subplot(6,3,i*3+2)
#     plt.imshow(heatmap)
#     if i == 0: plt.title('Heatmap')
#
#     plt.subplot(6,3,i*3+3)
#     plt.imshow(draw_img)
#     if i == 0: plt.title('Result')
#
# plt.savefig('output_images/bboxes_and_heat.png')
# plt.show()

video_output = 'test_video_res.mp4' # name of the video file generated by the vehicle detector
#clip1 = VideoFileClip("project_video.mp4") #original video file
clip1 = VideoFileClip("project_video.mp4") #original video file
white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(video_output, audio=False)
