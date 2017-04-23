import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn import preprocessing as pp
from sklearn import feature_extraction as fe

from utils import log


def onehot(y):
    Y = to_categorical(y)
    log('Y', Y.shape)
    return Y


def scale(x):
    scaler = pp.MinMaxScaler()
    scaler.fit(x)
    X = scaler.transform(x)
    X = pd.DataFrame(X)
    log('X', X.shape)
    return X.values, scaler


def label_encoder(labels, title=''):
    le = pp.LabelEncoder()
    le.fit(labels)
    log(title, len(le.classes_))
    return le


def word_encoder(words, title='', max_features=100, stop_words=[]):
    sw = fe.text.ENGLISH_STOP_WORDS.union(stop_words)
    cv = fe.text.TfidfVectorizer(
        stop_words=sw, max_features=max_features, use_idf=False, norm=None, binary=True)
    cv.fit(words)
    log(title, cv.get_feature_names())
    return cv
