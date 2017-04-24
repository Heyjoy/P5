# store
import pickle
import numpy as np
import collections

class datafield():
    def __init__(self):
        self.cars = []
        self.notcars = []
        self.TrainTestSplitSize = 0.2
        self.X_train, self.X_test, self.y_train, self.y_test = None,None,None,None

        # will save/load parameters with *.p file
        # if changed should delete the p file and runing  code again.
        self.color_space = 'YCrCb'  # RGB, HSV, LUV, HLS, YUV, YCrCb
        self.spatial_size = (8, 8)
        self.hist_bins = 16  # Number of histogram bins
        self.orient = 12  # HOG orientations
        self.pix_per_cell = 8  # HOG pixels per cell
        self.cell_per_block = 2  # HOG cells per block
        self.hog_channel = "ALL"  # 0, 1, 2, or "ALL"
        self.spatial_feat = True
        self.hist_feat = True
        self.hog_feat = True
        self.svc = None
        self.XScaler = None
        self.heat = None
        self.heatmaps = collections.deque(maxlen=10)
    def dataSave(self):
    #save current datafile parameters to .p file
        svc_pickle = {
        'svc':self.svc,
        #'X_train':X_train,
        #'X_test':X_test,
        #'y_train':y_train,
        #'y_test':y_test,
        'XScaler':self.XScaler,
        'color_space': self.color_space,
        'spatial_size': self.spatial_size,
        'hist_bins': self.hist_bins,
        'orient':self.orient,
        'pix_per_cell':self.pix_per_cell,
        'cell_per_block':self.cell_per_block,
        'hog_channel':self.hog_channel,
        'spatial_feat':self.spatial_feat,
        'hist_feat':self.hist_feat,
        'hog_feat':self.hog_feat
        }
        pickle.dump( svc_pickle, open( "SVC.p", "wb" ) )

    def dataLoad(self):
    #load the parameters from the .p file
        svc_pickle = pickle.load( open("SVC.p", "rb" ))
        #print(svc_pickle)
        self.color_space = svc_pickle['color_space']
        self.spatial_size = svc_pickle['spatial_size']
        self.hist_bins =svc_pickle['hist_bins']
        self.orient =svc_pickle['orient']
        self.pix_per_cell = svc_pickle['pix_per_cell']
        self.cell_per_block = svc_pickle['cell_per_block']
        self.hog_channel =svc_pickle['hog_channel']
        self.spatial_feat = svc_pickle['spatial_feat']
        self.hist_feat = svc_pickle['hist_feat']
        self.hog_feat = svc_pickle['hog_feat']
        self.svc = svc_pickle['svc']
        self.XScaler = svc_pickle['XScaler']

df = datafield()
