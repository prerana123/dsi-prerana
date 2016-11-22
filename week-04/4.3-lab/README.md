---
title: Regression, Cross Validation, & Regularization
type: lab
duration: "1:25"
creator:
    name: Jonathan Balaban
    city: Atlanta
---

# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Regression & Regularization Lab

## Introduction

In this lab you will practice performing regression while also applying cross-validation and regularization on a popular sample dataset. You'll be predicting the casual ridership of NYC Citibikes by applying these concepts to a bikeshare dataset.

## Requirements

Use the [starter code](./code/starter-code/starter-code.ipynb) to do the following:

1. Structure a new dataframe with features, create dummy features with pd.get_dummies(), and join dummies to data frame
2. Use Bikeshare dataset to create multiple linear models based on K-folds
3. Explain how K-Folds can address issues of overfitting by building multiple models and testing multiple models with different subsets of train and test data.
4. Build 3 different models: Regression, Lasso Regression, and Ridge Regression while also interpreting the MSE for each model.

## Deliverable

Functional, regularized, and cross validated models that predict casual ridership in NYC based on key features.

## Dataset

[The Citibike dataset can be found here](./assets/datasets/).

## Additional Resources

 - [Cross-validation with sklearn](http://scikit-learn.org/stable/modules/cross_validation.html)
 - [Ridge & Lasso Regression in Python](http://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/)
