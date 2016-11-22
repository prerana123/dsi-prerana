---
title: Logistic Regression and Regularization Scikit-Learn Lab
type: lab
duration: "1:25"
creator:
    name: Arun Ahuja
    city: NYC
---

# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Logistic Regression and Regularization Scikit-Learn Lab

## Exercise

Once, again we will attempt to predict when a website is 'evergreen'. However, this time we will use scikit-learn model _with regularization_. We will compare how the learned coefficients change under different regularization schemes. 

> From Week 4: Lab 2.2
These are websites that always relevant like recipes or reviews (as opposed to current events) This dataset comes from [stumbleupon](https://www.stumbleupon.com/), a web page recommender and was made available [here](https://www.kaggle.com/c/stumbleupon/download/train.tsv).  Webpages that are evergreen can always be recommended, however current events or seasonal stories can only be recommended at a specific time. For this reason, we'd like to identify websites that 'evergreen'. In this dataset, we have attributes about the website, for example, the text content, how many images or HTML elements it has, how many links it has etc.


In this lab, we will create models using the text of each website to evaluate whether or not the website is evergreen. Of course, not all of the words are relevant to this task. For example, almost every website will have the word 'link' somewhere - and it's unlikely that it effects whether or not the website is evergreen. How we can determine the most relevant features? Use regularization!

#### Requirements
- Practice using `scikit-learn` to run logistic regression to predict which sites are evergreen in the StumbleUpon dataset
- Compare `L1` or `Lasso` regularization with `L2` or `Ridge` regularization 
    - How do these effect the learned coefficients?
    - How do these effect the performance of the model using standard classification metrics?


#### Starter code

Here's a link to the [starter code](./code/starter-code/starter-code.ipynb)

> [Solution code found here](./code/solution-code/solution-code.ipynb)

#### Deliverable

- Build a logistic regression model in `scikit-learn`
- Examine the coefficients and performance for the different regularization schemes
- Create a writeup on the interpretation of findings including an executive summary with conclusions and next steps

