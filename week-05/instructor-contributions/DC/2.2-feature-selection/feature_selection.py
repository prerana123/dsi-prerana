# -*- coding: utf-8 -*-
"""
Created on Tue May 15 08:00:24 2016

@author: JosephNelson
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

iris.feature_names

model = LogisticRegression()
model.fit(X, y)

# The LogisticRegression class exposes an attribute called coef_. Let's have a look at it:
model.coef_

# According to the documentation this is:
# coef_ : array, shape (n_classes, n_features)
#    Coefficient of the features in the decision function.

# Let's display it in a nicer way:
coeffs = pd.DataFrame(model.coef_, columns = iris.feature_names, index =iris.target_names)
coeffs

# Question: Check: Can we conclude that petal length (cm) is the most significant feature to identify setosa ?


# let's normalize and redo

from sklearn.preprocessing import StandardScaler
X_norm =  StandardScaler().fit_transform(X)

model.fit(X_norm, y)

coeffs = pd.DataFrame(model.coef_, columns = iris.feature_names, index =iris.target_names)
coeffs

model


# visualize feature importance in a plot.

# first import matplotlib
% matplotlib inline
import matplotlib.pyplot as plt

# then create a figure and a plot
fig = plt.figure()
ax = fig.add_subplot(111)

# display the matrix
cax = ax.matshow(coeffs)

# http://matplotlib.org/api/figure_api.html

# add colorbars and tics
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(coeffs)
fig.colorbar(cax)
ax.set_xticklabels(['']+list(coeffs.columns), rotation=45)
ax.set_yticklabels(['']+list(coeffs.index))
