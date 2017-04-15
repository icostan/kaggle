import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn import preprocessing

from utils import log


def onehot(y):
    Y = to_categorical(y)
    log('Y', Y.shape)
    return Y


def scale(x):
    scaler = preprocessing.StandardScaler()
    scaler.fit(x)
    X = scaler.transform(x)
    X = pd.DataFrame(X)
    log('X', X.shape)
    return X.values, scaler


def label_encoder(labels, title=''):
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    log(title, len(le.classes_))
    return le
