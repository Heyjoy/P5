import glob
import numpy as np
from datafield import *
import extraction
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time
import pickle
from pathlib import Path
def datasetInit():
    images = glob.glob('dataset/*/*.png')

    for image in images:
        if 'image' in image or 'extra' in image:
            notcars.append(image)
        else:
            cars.append(image)

def datasetPrepare():
    carFeatures = extraction.extract_features(cars,color_space, spatial_size,
                            hist_bins, orient,
                            pix_per_cell, cell_per_block, hog_channel,
                            spatial_feat, hist_feat, hog_feat,False)
    notcarFeatures = extraction.extract_features(notcars,color_space, spatial_size,
                            hist_bins, orient,
                            pix_per_cell, cell_per_block, hog_channel,
                            spatial_feat, hist_feat, hog_feat,False)
    X = np.vstack((carFeatures, notcarFeatures)).astype(np.float64)
    y = np.hstack((np.ones(len(carFeatures)), np.zeros(len(notcarFeatures))))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TrainTestSplitSize)
    return X_train, X_test, y_train, y_test
def trainSVC(X_train, X_test, y_train, y_test):
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    print('Train Accuracy:', round(svc.score(X_train, y_train), 3))
    print('Test Accuracy: ', round(svc.score(X_test, y_test), 3))
    svc_pickle = {}
    svc_pickle["svc"] = svc
    pickle.dump( svc_pickle, open( "SVC.p", "wb" ) )

def getSVC():
    trainedSVCPath = Path("./SVC.p")
    if trainedSVCPath.is_file():
        print("we have trained SVC")
    else:
        print("we have no trained SVC, now training....")
        print("please runing the code later again")
        datasetInit()
        print(len(cars),len(notcars))
        X_train, X_test, y_train, y_test = datasetPrepare()
        trainSVC(X_train, X_test, y_train, y_test)
