import os
import numpy as np
import cv2
from sklearn import preprocessing
from utils import log
from keras.preprocessing import image

INPUT_FOLDER = 'input/'
TRAIN_FOLDER = INPUT_FOLDER + 'train/'
TEST_FOLDER = INPUT_FOLDER + 'test_stg2/'

# ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
SIZE = 128

def load_train_data(categories, size=SIZE, verbose=False):
    x = []
    y = []
    log('Status', 'Processing... ' + str(categories))
    for t in categories:
        folder = TRAIN_FOLDER + t
        files = os.listdir(folder)
        log(t, str(len(files)) + ' files')
        for filename in files:
            img = load_image(folder + '/' + filename, expand_dims=False)
            x.append(img)
            y.append(t)
    log('Status', 'DONE')

    X = normalize(np.array(x))
    log('X shape', X.shape)
    log('X size', bytesto(X.nbytes, 'k'))

    Y = preprocessing.LabelEncoder().fit_transform(np.array(y))
    log('Y shape', Y.shape)
    log('Y size', bytesto(Y.nbytes, 'k'))

    return X, Y

def load_image(path, size=SIZE, expand_dims=True):
    img = image.load_img(path, target_size=(size, size))
    X = image.img_to_array(img)
    if expand_dims:
        X = np.expand_dims(X, axis=0)
    return X

def load_test_data():
    t = []
    files = os.listdir(TEST_FOLDER)
    log('Status', 'Processing... ' + str(len(files)) + ' files')
    for filename in files:
        path = TEST_FOLDER + '/' + filename
        img = load_image(path, expand_dims=False)
        t.append(img)
    log('Status', 'DONE')
    T = normalize(np.array(t))
    log('Shape', T.shape)
    log('Size', bytesto(t.nbytes, 'k'))
    return T, files
