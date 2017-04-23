from utils import log
from sklearn import feature_selection as fs
import pandas as pd


def threshold_filter(x, feature_names):
    selector = fs.VarianceThreshold(threshold=0.8)
    xv = selector.fit_transform(x)
    log('Threshold', selected_features(selector, feature_names))
    log('X', xv.shape)
    return xv, selector


def percentile_filter(x, y, feature_names, percentile=20):
    selector = fs.SelectPercentile(fs.chi2, percentile=percentile)
    xp = selector.fit_transform(x, y)
    features = selected_features(selector, feature_names)
    log('Percentile', features)
    log('X', xp.shape)
    return pd.DataFrame(xp, columns=features, index=x.index)


def kbest_filter(x, y, feature_names, k=10):
    selector = fs.SelectKBest(fs.chi2, k=k)
    xp = selector.fit_transform(x, y)
    features = selected_features(selector, feature_names)
    # log('KBest', features)
    log('X', xp.shape)
    return pd.DataFrame(xp, columns=features, index=x.index), selector


def selected_features(selector, feature_names):
    idx = selector.get_support(feature_names)
    return list(map(lambda i: feature_names[i], idx))
