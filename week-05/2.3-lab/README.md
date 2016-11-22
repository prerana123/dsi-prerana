---
title: Project Pipeline
type: lab
duration: "1:25"
creator:
    name: Francesco Mosconi
    city: SF
---

# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project Pipeline

Setup pre-processing / pipelines for Project 5


## Introduction

> ***Note:*** _This can be a pair programming activity or done independently._

In this lab you will practice organizing data processing steps and documenting each step well so the entire process is clear and reproducible.

In particular we will create pipelines for data processing on the [Titanic dataset]('../../assets/datasets/train.csv').

The dataset is a list of passengers. The second column of the dataset is a “label” for each person indicating whether that person survived (1) or did not survive (0). Here is the Kaggle page with more information on the dataset:

http://www.kaggle.com/c/titanic-gettingStarted/data

As a first step load the data in a pandas dataframe.

## Exercise

#### Requirements

- Import the data
- Impute missing data
- Deal with categorical variables
- Deal with boolean features
- Deal with scale
- Combine features

**Bonus:**
- Family feature

#### Starter code

[Starter code](code/starter-code/starter-code-2_3.ipynb)

> [Solution code](code/solution-code/solution-code-2_3.ipynb)


## Additional Resources

- [Scikit Learn Pipelines Documentation](http://scikit-learn.org/stable/modules/pipeline.html#pipeline)
