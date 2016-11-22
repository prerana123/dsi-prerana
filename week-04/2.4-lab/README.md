---
title: Logistic Regression Scikit-Learn Lab
type: lab
duration: "1:25"
creator:
    name: Arun Ahuja
    city: NYC
---

# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Logistic Regression Scikit-Learn Lab

## Exercise

Once, again we will attempt to predict when a website is 'evergreen'. However, this time we will use scikit-learn. Additionally, we evaluate our model using standard classification metrics such as precision, recall and AUC.


> From Week 4: Lab 2.2
These are websites that always relevant like recipes or reviews (as opposed to current events) This dataset comes from [stumbleupon](https://www.stumbleupon.com/), a web page recommender and was made available [here](https://www.kaggle.com/c/stumbleupon/download/train.tsv).  Webpages that are evergreen can always be recommended, however current events or seasonal stories can only be recommended at a specific time. For this reason, we'd like to identify websites that 'evergreen'. In this dataset, we have attributes about the website, for example, the text content, how many images or HTML elements it has, how many links it has etc.


#### Requirements
- Practice using `scikit-learn` to run logistic regression to predict which sites are evergreen in the StumbleUpon dataset
- Evaluate the model using classification metrics such as: accuracy, precision, recall and AUC
- Introduce new features to the model and see if it makes a difference

#### Starter code

Here's a link to the [starter code](./code/starter-code/starter-code.ipynb)

> [Solution code found here](./code/solution-code/solution-code.ipynb)

#### Deliverable

- Build a logistic regression model in `scikit-learn`
- Examine the performance for each feature added to the model
- Create a writeup on the interpretation of findings including an executive summary with conclusions and next steps

