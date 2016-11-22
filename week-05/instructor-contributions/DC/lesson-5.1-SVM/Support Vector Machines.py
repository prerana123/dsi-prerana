# -*- coding: utf-8 -*-
"""
Created on Tue May 17 08:41:59 2016

@author: JosephNelson
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

model = SVC(kernel='linear')
model.fit(X, y)

'''
Notice that the SVC class has several parameters. In particular we are concerned with two:

C: penalty parameter of the error term (regularization)
kernel: the type of kernel used (linear, poly, rbf, sigmoid, precomputed or a callable.)

Notes from the documentation:

In the current implementation the fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset with more than a couple of 10000 samples.
The multi-class support is handled according to a one-vs-one scheme.
As usual we can calculate the cross validated score to judge the quality of the model.

'''

from sklearn.cross_validation import cross_val_score

cvscores = cross_val_score(model, X, y, cv = 5, n_jobs=-1)
print "CV score: {:.3} +/- {:.3}".format(cvscores.mean(), cvscores.std())


# Gridsearch

from sklearn.grid_search import GridSearchCV
parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 3, 10]}
clf = GridSearchCV(model, parameters, n_jobs=-1)
clf.fit(X, y)
clf.best_estimator_

'''
Pros:

Very powerful, good performance
Can be used for anomaly detection (one-class SVM)
Cons:

Can get very hard to train with lots of data
Prone to overfit (need regularization)
Black box
'''
