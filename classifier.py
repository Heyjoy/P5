import glob
import numpy as np
import extraction
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import time
from pathlib import Path
import pickle
from datafield import *

def datasetInit():
    #images = glob.glob('dataset/*/*.png')
    df.cars = glob.glob('dataset/vehicles/*/*.png')
    df.notcars = glob.glob('dataset/non-vehicles/*/*.png')
    print("import carsample:{} non-car sample:{}".format(len(df.cars),len(df.notcars)))
    #for image in images:
    #    if 'image' in image or 'extra' in image:
    #        df.notcars.append(image)
    #    else:
    #        df.cars.append(image)

def datasetPrepare():
    carFeatures = extraction.extract_features(df.cars,df.color_space, df.spatial_size,
                            df.hist_bins, df.orient,
                            df.pix_per_cell, df.cell_per_block, df.hog_channel,
                            df.spatial_feat, df.hist_feat, df.hog_feat,False)
    notcarFeatures = extraction.extract_features(df.notcars,df.color_space, df.spatial_size,
                            df.hist_bins, df.orient,
                            df.pix_per_cell, df.cell_per_block, df.hog_channel,
                            df.spatial_feat, df.hist_feat, df.hog_feat,False)
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
    # Check the training time for the SVC
    t=time.time()
    svc.fit(df.X_train, df.y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(df.X_test, df.y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(df.X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', df.y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    # update to the datafield
    df.svc = svc
def dataInit():
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
