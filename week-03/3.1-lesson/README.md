---
title: Feature Scaling
duration: "1:25"
creator:
    name: Marc Harper
    city: LA
---

# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Feature Scaling
Week 3 | Lesson 3.1

### LEARNING OBJECTIVES
*After this lesson, you will be able to:*
- Use the scikit-learn preprocessing module to normalize the data in various ways

### STUDENT PRE-WORK
*Before this lesson, you should already be able to:*
- Load and manipulate data with Pandas
- Fit models with sklearn

### INSTRUCTOR PREP
*Before this lesson, instructors will need to:*
- Read in / Review any dataset(s) & starter/solution code
- Generate a brief slide deck
- Prepare any specific materials
- Provide students with additional resources

### LESSON GUIDE
| TIMING  | TYPE  | TOPIC  |
|:-:|---|---|
| 5 min  | [Opening](#opening)  | Discussion  |
| 15 min  | [Introduction](#introduction)   | Feature Scaling  |
| 20 min  | [Demo](#demo)  | Scaling in Python  |
| 15 min  | [Guided Practice](#guided-practice<a name="opening"></a>)  | Normalization  |
| 35 min  | [Independent Practice](#ind-practice)  | Scaling and Linear Regression |
| 5 min  | [Conclusion](#conclusion)  | Review, Recap |

---

<a name="opening"></a>
## Opening (5 mins)
- Review prior labs/homework, upcoming projects, or exit tickets, when applicable
- Review lesson objectives
- Discuss real world relevance of these topics
- Relate topics to the [Data Science Workflow](https://drive.google.com/file/d/0Bx2SHQGVqWasOGY4dE95OFVvZjQ/view?usp=sharing) - i.e. are these concepts typically used to acquire, parse, clean, mine, refine, model, present, or deploy?


<a name="introduction"></a>
## Introduction: Feature Scaling (15-20 mins)

When working with new data sets we always need to process the data. As we've
seen it's usually necessary to convert strings to numbers, handle date formats,
and toss out bad data points. It's also often necessary to scale our data, and
it is rarely not a good idea.

### Why scale data?

There are a number of good reasons why we scale our data:
* to handle disparities in units
* because many machine learning models require scaling
* it can speed up gradient descent

The [sci-kit learn documentation](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) lays it out pretty clearly::

Standardization of a dataset is a common requirement for many machine
learning estimators: they might behave badly if the individual feature do
not more or less look like standard normally distributed data
(e.g. Gaussian with 0 mean and unit variance).

The reason we scale for gradient descent is to prevent major differences in the
steps on different axis to be widely different. This makes it difficult to
find a good learning rate since once that is too small will take a long time
to move around in the direction of a larger-scale feature, and a learning rate
that is too large will not have good resolution on a smaller-scale feature.

The good news is that it's rarely a bad idea to scale your data. So it's a good
practice to apply consistently, and to master early in your progression as a
data scientist.

### How to we scale our data?

Typically we scale features in one of a few standard ways. For example, a
common method called _standardization_ takes a feature and rescales it to
have mean zero and variance 1, like a _standard_ normal distribution. We do this
by computing the mean and standard deviation, and then transform data as so:

```
x' = (x - mean) / std_dev
```

Standardization is sometimes called centering. Another common method is called
_Min-Max Scaling_ or simply _rescaling_. In this
case we rescale our data to fit into an interval `(min, max)` by transforming:

```
x' = (x - min) / (max - min)
```

_Normalization_ is another scaling method that you may have seen before, and it
involves dividing the data in a feature by the sum of all the features. It you
know what a unit vector is then you've seen normalization before, in which
case you divided by the magnitude of a vector (the square root of the sum of
the squares).

### Interpretation of Coefficients

By scaling our data we change the interpretations of the coefficients. In a regression of the form
```
y = a_0 + a_1 x_1 + ... + a_n x_n
```

recall that we interpret `a_i` as the amount that `y` changes when `x_i` changes by one unit.
Scaling the data can change the interpretation of some of the coefficients. If we normalize the
data, we've changed the scale, and hence the interpretation, since a change of one unit in the
scaled variable likely won't produce the same effect (i.e. it's a slope).

For example, if we center (standardize) our predictors then the constant term `a_0`
is the mean of `y`. If we center the `y` values then `a_0` will be zero. The other `a_i`
now tell us how many standard deviations `y` will change when the variable `x_i` changes
by one standard deviation, since we've normalized all the standard deviations to 1. Such
coefficients are often called [standardized coefficients](https://en.wikipedia.org/wiki/Standardized_coefficient).

Standardizing the coefficients can help determine which independent variable `x_i` has the
greater impact on the dependent variable `y` because we've taken away the impact of
different scalings and units of the `x_i`.

**Important Notes**:
* Note that scaling won't improve the statistics of a linear regression.
* The various `x_i` and `y` may not be normal
distributions just because we've standardized them, so when we say that a change of one
standard deviation, we mean in the variables actual distribution, which can be very
different for non-normal distributions.
* If there are dependencies among the predictors then interpretations become more complex.
For example, with a quadratic regression `y = a_0 + a_1 x + a_2 x^2`, it is not possible to 
have a one unit change in `x` without also changing `x^2`, so we cannot interpret the coefficients
in the usual way regardless if we scale the data.

In the case that there are interacting terms, such as `x` and `x^2` or products like
`x_1 x_2` and `x_2 x_3`, the proper interpretation of a coefficient is the response of `y`
to a change in the associated predictor while allowing for the contributions of 
the other predictors (which may also change if there are dependencies). There is a
good discussion with examples [here](https://umassmed.edu/uploadedFiles/QHS/Content/Making%20Sense_12Jan13.pdf)
if you'd like to dig deeper.


**Check**: Why do we scale data?


<a name="demo"></a>
## Demo: Scaling in Python (20 mins)

Use the [starter code](./code/starter-code/Feature-Scaling-Starter.ipynb) to walk through the demo
of different scaling methods.

<a name="guided-practice"></a>
## Guided Practice: Normalization (15 mins)

Practice scaling by normalization using pure Python and scikit-learn with the
[starter code](./code/starter-code/Feature-Scaling-Starter.ipynb).
- Apply L1 and L2 normalization using python (5-10 mins)
- Apply L1 and L2 normalization using scikit-learn (5-10 mins)

> [Solution code](./code/solution-code/Feature-Scaling-Solutions.ipynb)

<a name="ind-practice"></a>
## Independent Practice: Scaling and Linear Regression (30 minutes)

- Practice scaling and linear fits. Does normalization affect any of your models? (10-20 mins)
- Try some regularized models. Does scaling have a significant effect? (10 mins)
- Try some other models from scikit-learn, such as a [SGDRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html).
It's ok if you are unfamiliar with the model, just follow the example code
and explore the fit and the effect of scaling. (10 mins)
- **Bonus**: try a few extra models like a [support vector machine](http://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html). What do you think
about the goodness of fit? Scaling is _required_ for this model.

> Scaling doesn't affect linear regression typically. The Bonus exercise asks
students to choose another model that scaling is necessary for. Students may
need a little guidance but really the need to only change one line (the model).

> If students don't make it to the bonus exercise, take a few minutes at the
end to show them that scaling does sometimes matter for e.g. and SGDRegressor.
The support vector machine is a great fit!

> [Solution code](./code/solution-code/Feature-Scaling-Solutions.ipynb)

**Check:** Does scaling affect linear regression?


<a name="conclusion"></a>
## Conclusion (# mins)
Let's review! Discuss a few of the reasons that scaling is important:

>
* To handle disparities in units.
* Because many machine learning models require scaling
* It can speed up gradient descent

***


### ADDITIONAL RESOURCES

- [Feature scaling](http://sebastianraschka.com/Articles/2014_about_feature_scaling.html#z-score-standardization-or-min-max-scaling)
the wine dataset.
- [Some examples of regression](http://facweb.cs.depaul.edu/mobasher/classes/CSC478/Notes/IPython%20Notebook%20-%20Regression.html) with the Boston dataset.
