---
title: Intro to Classification
duration: "1:25"
creator:
    name: Faith / Jon
    city: Washington / ATL
---

# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Intro to Classification
Week 4 | Lesson 1.1

### LEARNING OBJECTIVES
*After this lesson, you will be able to:*
- Define and give examples of classification
- Explain kNN algorithm
- Build a K-Nearest Neighbors using the scikit-learn library

### STUDENT PRE-WORK
*Before this lesson, you should already be able to:*
- Run Python on your laptop
- Have a basic understanding of the key concepts of Python programming
- Experience with the scikit-learn library
- Basic experience with machine learning models (linear regression and cross validation)
- Definition of Euclidean distance

### INSTRUCTOR PREP
*Before this lesson, instructors will need to:*
- Review Training Package
- Copy and modify the lesson slide deck as required
- Read through datasets and starter/solution code
- Add to the "Additional Resources" section for this lesson


### LESSON GUIDE
| TIMING  | TYPE  | TOPIC  |
|:-:|---|---|
| 5 min  | [Opening](#opening)  | Discuss lesson objectives |
| 10 min  | [Introduction](#introduction)   | Description of Classification  |
| 20 min  | [Demo](#demo)  | How kNN logically works   |
| 25 min  | [Guided Practice](#guided-practice) | Practice kNN and credit default |
| 25 min  | [Independent Practice](#ind-practice)  |  Training set to classify an unknown case  |
| 5 min  | [Conclusion](#conclusion)  | Review / Recap |

---

<a name="opening"></a>
## Opening (5 mins)
- Review current lesson objectives
- Reference general course content or topics (e.g. code or concepts that have been used across multiple lessons)
- Include Hook / Real-world Relevance (why the content from this lesson is useful or important)

> **Check:** Ask students to define, explain, or recall prior concepts or tools.

<a name="introduction"></a>
## Introduction: Topic (10 mins)

Classification is a machine learning technique to assign a label or category on new instances of data from already observed past data whose label or category is known. There are a number of different classification techniques including kNN (k-Nearest Neighbors).

While in regression we are attempting to predict a continuous variable from input data, in classification we are attempting to predict a _class_ or _category_ from the data. Most classification problems are binary, in that we predict whether one of two possibilities will happen. However, we could predict one of many possible classes, which is known as _multi-class_ classification.

For example, we might build a machine learning model to detect spam emails. By analyzing the content of emails and ‘classify’ them as either spam or legitimate emails. This is an example of a _binary_ classification problem. Another example, we might have a number of species of flowers and would like to know what specie of flower is from a dataset of known flowers. This may be an example of a _multi-class_ classification problem.

Classification is a machine learning problem for solving a set value given the knowledge we have about that value. Many classification problems boil down to a binary problem. For example, with patient data, one could be working on solving a treatment problem for smokers but first we need to know if their medical history suggests, or is predictive, of whether the patient is a smoker or not.

Many problems don't appear to be binary at first glance, but with a limited set of variables, you can usually boil your model down to a boolean value. For instance, what if you are predicting whether an image pixel will be red or blue? In analyzing the image, we could check whether a pixel "is red" or "is not red." Binary classification is the simplest form of classification, though classification problems can certainly be wrapped around multiple class labels.

#### K-Nearest Neighbors (kNN)

K-Nearest Neighbors (kNN) classification algorithm is one of the simplest to perform and comprehend. It utilizes proximity to known data points with known classifications.

K nearest neighbors stores all available cases and classifies new cases based on a similarity measure KNN has been used in statistical estimation and pattern recognition already in the beginning of 1970’s as a non-parametric technique.

A case is classified by a majority vote of its neighbors, with the case being assigned to the class most common amongst its K nearest neighbors measured by a distance function. If K = 1, then the case is simply assigned to the class of its nearest neighbor.

Choosing the optimal value for K is best done by first inspecting the data. In general, a large K value is more precise as it reduces the overall noise but there is no guarantee. Cross-validation is another way to retrospectively determine a good K value by using an independent dataset to validate the K value. Historically, the optimal K for most datasets has been between 3-10. That produces much better results than 1NN.

<a name="demo"></a>
## Demo: How kNN logically works (20 mins)

Here is some pseudo-code to demonstrate how kNN logically works:

```python
procedure KNN(x)
  begin
    looping through all known data points in training data, find the closest k points to x
    assign f(x) = majority classification among the k closest points
  end
```
Consider the following data concerning credit default. Age and Loan are two numerical variables (predictors) and Default is the target.

![](./assets/images/graph1.png)

We can now use the training set to classify an unknown case (Age=48 and Loan=$142,000) using Euclidean distance. If K=1 then the nearest neighbor is the last case in the training set with Default=Y.

Remember, the magnitudes of the variables matter so if we don't standardize the dataset, the `Loan` variable will contribute more to the distance!

![](./assets/images/graph.week.4class1-2.png)

With K=3, there are two Default=Y and one Default=N out of three closest neighbors. The prediction for the unknown case is again Default=Y.

While the algorithm is relatively simple, there are many caveats and important details. Note that we need to define what _closest_ means in terms of each data point. Typically, this means that we need to define a _distance measure_ between two datapoints. Euclidean distance is the most common, but there are alternatives, i.e. cosine distance or Jaccard distance.

Also note, that because we need to compute distance between datapoints the magnitudes of of variables matter. Variables with larger magnitudes will appear far apart, so we must remember to scale or standardize the dataset.

Lastly, note that every variable is used no matter what. If we are attempting to classify loans as those likely to default, every variable would contribute, even if they had no relation to the label! For example, if we used variables such as: income (useful) and height (not useful), distance in height would contribute just as much to the classification as distance in income!

From [Cathy O'Neils mathbabe blog, she referred to KNN as **dangerously simple**](https://mathbabe.org/2013/04/04/k-nearest-neighbors-dangerously-simple/)

> The devil is all in the detail of what you mean by close. And to make things trickier, as in easier to be deceptively easy, there are default choices you could make (and which you would make) which would probably be totally stupid. Namely, the raw numbers, and Euclidean distance.
So, for example, say your customer attributes were: age, salary, and number of previous visits to your website. 
> In other words, because you’re including the raw salary numbers, you are thinking of Monica, who is 22 and new to the site, as close to a 75-year old who comes to the site a lot. The salary, being of a much larger scale, is totally dominating the distance calculation. You might as well have only that one attribute and scrap the others.

>Note: you would not necessarily think about this problem if you were just pressing a big button on a dashboard called “k-NN me!”

>Of course, it gets trickier. Even if you measured salary in thousands (so Monica would now be given the attribution vector (22.0, 55.0, 0.0)) you still don’t know if that’s the right scaling. In fact, if you think about it, the algorithm’s results completely depends on how you scale these numbers, and there’s almost no way to reasonably visualize it even, to do it by hand, if you have more than 4 attributes.

>Another problem is redundancy – if you have a bunch of attributes that are essentially redundant, i.e. that are highly correlated to each other, then including them all is tantamount to multiplying the scale of that factor.

#### Advantages:
- Simple, easy to explain
- Non-linear
    - While linear regression and other linear models can all describe linear trends 
    - (as more of the input happens the output can only change in  one direction)
        - More income -> less likely to default
        - Higher education -> higher income
    - Non-linear trends are more common
        - From 0 - 16, income is likely constantly 0
        - From 16 - 65, income increases, perhaps plateau-ing
        - From 65+ (or earlier), income likely falls back!
    - KNN can pick-up these non-linear trends

#### Disadvantages:
- No learning or pattern extraction
    - KNN **always** relies on the input dataset to make future predictions.
    - It does not extract any patterns (i.e. coefficients in linear regression) that be used to generalize what is happening or answer questions about the importance of different variables.
    - All variables are used, even irrelevant ones. No feature selection (unlike linear regression which would set coefficients to 0)
- Large memory footprint and slow predictions
    - To compute distances we need to store _every_ training datapoint
    - This means lots of memory to store the datapoints and a long computation (pairwise distance) to decide what the label or category is for a new datapoint.


<a name="guided-practice"></a>
## Guided Practice: Demo: Basic kNN Python example code (25 mins)

```python
>>> X = [[0], [1], [2], [3]]
>>> y = [0, 0, 1, 1]
>>> from sklearn.neighbors import KNeighborsClassifier
>>> neigh = KNeighborsClassifier(n_neighbors=3)
>>> neigh.fit(X, y)
KNeighborsClassifier(...)
>>> print(neigh.predict([[1.1]]))
[0]
>>> print(neigh.predict_proba([[0.9]]))
[[ 0.66666667  0.33333333]]

```

<a name="ind-practice"></a>
## Independent Practice: KNN & Breast Cancer Dataset (25 minutes)

Using the Breast Cancer dataset, perform a simple kNN classification on whether a tumor is malignant or benign.

Refer to the [Starter Code](./code/starter-code/starter-code.ipynb) to begin.

> **Check: Instructors should review student understanding of model score & interpretation of confusion matrix**

> Here is the [solution code](./code/solution-code/solution-code.ipynb)

<a name="conclusion"></a>
## Conclusion (5 mins)
- What are class labels? What does it mean to classify?
- When do you perform classification ?
- How does the kNN algorithm work?
- How do you define: accuracy, misclassification?

***

### ADDITIONAL RESOURCES
- [kNN Tutorial](http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/)
- [K-Nearest Neighbors: dangerously simple](https://mathbabe.org/2013/04/04/k-nearest-neighbors-dangerously-simple/)
- [KNN Classification & Assumptions](https://saravananthirumuruganathan.wordpress.com/2010/05/17/a-detailed-introduction-to-k-nearest-neighbor-knn-algorithm/)
- [How KNN Works](https://saravananthirumuruganathan.wordpress.com/2010/05/17/a-detailed-introduction-to-k-nearest-neighbor-knn-algorithm/)
