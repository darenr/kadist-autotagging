#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn import metrics


#
# metrics
#

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None, label_encode=True):
    '''
    Compute the Hamming score (label-based accuracy) for multi-label predictions
    takes an array of array of strings
    '''

    if label_encode:
        arr = []
        for t in y_true + y_pred:
            arr.extend(t)

        e = preprocessing.LabelEncoder()
        e.fit(arr)

        _y_true = np.array([e.transform(x) for x in y_true])
        _y_pred = np.array([e.transform(x) for x in y_pred])
    else:
        _y_true = y_true
        _y_pred = y_pred

    acc_list = []
    for i in range(_y_true.shape[0]):
        set_true = set(np.where(_y_true[i])[0])
        set_pred = set(np.where(_y_pred[i])[0])
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / float(len(set_true.union(set_pred)))
        acc_list.append(tmp_a)

    return np.mean(acc_list)
