import glob
import numpy as np
from datafield import *
import extraction
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import time
import pickle
from pathlib import Path
def datasetInit():
    images = glob.glob('dataset/*/*.png')

    for image in images:
        if 'image' in image or 'extra' in image:
            df.notcars.append(image)
        else:
            df.cars.append(image)

def datasetPrepare():
    print("{},{}".format(len(df.cars),len(df.notcars)))
    carFeatures = extraction.extract_features(df.cars,df.color_space, df.spatial_size,
                            df.hist_bins, df.orient,
                            df.pix_per_cell, df.cell_per_block, df.hog_channel,
                            df.spatial_feat, df.hist_feat, df.hog_feat,False)
    notcarFeatures = extraction.extract_features(df.notcars,df.color_space, df.spatial_size,
                            df.hist_bins, df.orient,
                            df.pix_per_cell, df.cell_per_block, df.hog_channel,
                            df.spatial_feat, df.hist_feat, df.hog_feat,False)
    print("car features:")
    print(carFeatures)
    print("no car feature")
    print(notcarFeatures)

    X = np.vstack((carFeatures, notcarFeatures)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    y = np.hstack((np.ones(len(carFeatures)), np.zeros(len(notcarFeatures))))
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    #update to the datafield
    df.XScaler = X_scaler
    df.X_train, df.X_test, df.y_train, df.y_test = train_test_split(scaled_X, y,
                    test_size= df.TrainTestSplitSize,random_state=rand_state)
def trainSVC():
    svc = LinearSVC()
    svc.fit(df.X_train, df.y_train)
    print('Train Accuracy:', round(svc.score(df.X_train, df.y_train), 3))
    print('Test Accuracy: ', round(svc.score(df.X_test, df.y_test), 3))


def getSVC():
    trainedSVCPath = Path("SVC.p")
    if trainedSVCPath.is_file():
        print("we have trained SVC, now loading...")
        df.dataLoad()
    else:
        print("we have no trained SVC, now training....")
        print("please runing the code later again")
        datasetInit()
        datasetPrepare()
        trainSVC()
        df.dataSave()
