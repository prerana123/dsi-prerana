---
title: Intro to Modeling - Linear Regression
duration: "1:25"
creator:
    name: Marc Harper
    city: LA
---

# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Intro to Modeling: Linear Regression
Week 3 | Lesson 1.1

### LEARNING OBJECTIVES
*After this lesson, you will be able to:*
- Define the terms: modeling, prediction
- Understand the best fit line of a set of data
- Find the best fit line by hand

### STUDENT PRE-WORK
*Before this lesson, you should already be able to:*
- Be able to use python for simple calculations

### INSTRUCTOR PREP
*Before this lesson, instructors will need to:*
- Read in / Review any dataset(s) & starter/solution code
- Generate a brief slide deck
- Prepare any specific materials
- Provide students with additional resources

> Instructors: If you need to compute any best fit lines, the solution code for this lesson computes best fit lines and coefficients and is easy to modify!

### LESSON GUIDE
| TIMING  | TYPE  | TOPIC  |
|:-:|---|---|
| 5 min  | [Opening](#opening)  | Discussion |
| 15 min  | [Introduction](#introduction)   | Intro to Modeling with Linear Regression  |
| 10 min  | [Demo](#demo)  | Define your own best fit lines |
| 20 min  | [Guided Practice](#guided-practice<a name="opening"></a>)  | Define Best Fit Lines  |
| 10 min  | [Demo](#demo)  | Equations for Best Fit |
| 10 min  | [Guided Practice](#guided-practice<a name="opening"></a>)  | Compute the true best fit line  |
| 15 min  | [Independent Practice](#ind-practice)  | Compute the best fit on a final data set  |
| 5 min  | [Conclusion](#conclusion)  | Review, Recap |

---

<a name="opening"></a>
## Opening (5 mins)
- Review prior labs/homework, upcoming projects, or exit tickets, when applicable
- Review lesson objectives
- Discuss real world relevance of these topics
- Relate topics to the [Data Science Workflow](https://drive.google.com/file/d/0Bx2SHQGVqWasOGY4dE95OFVvZjQ/view?usp=sharing) - i.e. are these concepts typically used to acquire, parse, clean, mine, refine, model, present, or deploy?


<a name="introduction"></a>
## Introduction: Intro to Modeling with Linear Regression (15 mins)

### Modeling

* All humans naturally model the world around them.

For example, you know when traffic is heaviest on the streets and intersections near your home. Over time
your observations have built up a mental dataset and a mental model that helps
you predict what traffic will be like at various times and locations. You
probably use this mental model to help plan your days, predict arrival times,
and many other tasks.

> Have the class think of other examples. How about your weekly grocery list -- you've built up a model of how much of each item you use in a week, and when you'll need more.

* As data scientists we make the relationships between quantities precise using data and mathematical and statistical structures
* This process is called _modeling_
* _linear regression_ is an extremely commonly tool for modeling relationships.

For example, if you
drink a glass of orange juice every day and the bottle holds 8 glasses, then we
know how much of a bottle you drink in `d` days is
`j = (1 / 8) * d`. From this we can predict that you will consume 7/8-ths of a
bottle every week, 8 bottles every seven weeks, and so on.

* When you drive at a constant speed `s` for `t` hours, the distance you travel is
`d = st`.
* The quantities of distance and time are in a linear relationship
because the graph is a line with slope (rate) `s`.
* This equation is a model
between the distance travelled and the time travelled.

Because we use rates to commonly understand natural processes, linear
relationships are ubiquitous in our lives and our data. We don't always know
how two quantities are related. Linear regression is a way to determine the
relationship from a set of data.

* Models are relationships between quantities
* Linear regression is a method to determine the coefficients of linear
relationships

### Prediction

* With a model we can make _predictions_.

For example, if I know that you are driving
35 miles per hour for 2 hours then I predict that you will travel 70 miles.

In general our models are not so precise if the relationship between our quantities
are not perfect, but we can still make a reasonable guess using our model.

* Predictions can be very valuable even if they are not always exactly right.

For example, if you own a movie theatre, you need need to make sure you have enough
staff on nights and weekends to handle the increased number of patrons. A model
to predict how many movie-goers will show up on a given night will help you
predict the amount of employees that you'll need -- enough to handle the customers
but not too many so that you keep labor costs reasonable. Too few employees and
you won't be able to serve all the customers; too many and you've spent too much
on labor. The better the estimate the better your business runs, and even if
the estimate is off a bit you are still more prepared than if you had no
model at all.

Similarly, other models might predict how much inventory to stock, how much
a component or raw material will cost during different times of year, the
percentage of voters that will show up in various weather conditions, and just
about anything else.

* With models we can make predictions
* Good predictions are extremely valuable for a wide variety of purposes

**Check:** Ask students to talk through other scenarios in which predictions are
valuable from their work experience.

### Linear Regression

What is linear regression? Simply put it is a method of determining the coefficients
of a linear combination of variables:

```
y = a_0 x_0 + a_1 x_1 + ... + a_n x_n
```

To use linear regression, the variables themselves need not be linear, just the
relationships between the quantities. So we could, for example, fit a target
variable `y` to a polynomial:

```
y = a_0 + a_1 x + a_2 x^2
```

In this case we will have to supply both `x` and `x^2` for every data point. For
example, the trajectory of a ball thrown through the air forms a parabola with equation
`d = d_0 + v t + 0.5 a t^2`
where `d` is the distance travelled, `v` is the velocity, and `a` is
the acceleration due to gravity (9.8 m / s^2 on the surface of Earth). We could
determine the coefficients `v` and `a` from a set of data using linear regression.

## Demo: Linear Regression By Eye (10 mins)

> Instructor Note: Draw on the white board or display a data set of a randomly
generated data set with some noise. (See the assets folder for a sample). Plot
a best fit line by eye and measure two points on the line. Calculate the slope
and use the slope intercept equation to find the "best-fit line":

> A nice way to do this is to simply use the relationship y=x^2, starting with the
points (0, 0) and (1, 1). Do the best fit line calculation. Next add the points
(2, 4) and (3, 9) to the plot and redo the fit using the points (0, 0) and (3,9).

> Later you can calculate the actual best fit line using the same three points,
showing that the best fit line doesn't necessarily go through any of the data
points.

```
m = (y_1 - y_0) / (x_1 - x_0)

y - y_0 = m (x - x_0)
```

## Guided Practice: Define your own best fit lines (20 minutes)

> Now provide another small data set and have the class break into small groups.
Each group should determine their own best fit line. After about 15 minutes, or
when all the groups have finished, compare the coefficients and lines.

> Instructor Note: The point of the exercise is to show that each group will
not come up with the exact same best fit line (a few outliers should guarantee
this). This paves the way for giving the mathematical best approach.

> The sample dataset has a best fit line of `y = 0.9 + 2.1 x` (rounded coefficients). Likely students
will conclude that the model is approximately `y = 1 + 2 x` or some other variation.

Sample data:

(1, 3.0)
(2, 5.1)
(3, 7.1)
(4, 9.4)
(5, 11.5)
(6, 13.5)
(7, 15.6)
(8, 17.7)
(9, 19.8)


## Demo: Equations for Best Fit (10 minutes)

>Instructor Note: Present the equations for the best fit line for the least
squares method for the
[simple linear regression model](https://en.wikipedia.org/wiki/Simple_linear_regression).
Explain that this model minimizes the vertical squared errors of the best fit line.
If time and the class is technical adept enough, describe the squared error as the
sum of the *residuals*, the difference between the prediction and the true value
squared, summed over all points.

The equations are:

![Best fit coefficients](https://upload.wikimedia.org/math/e/e/d/eed68731d1230938d457c576deee1bcf.png)

> If you have time, complete the best fit for the data points listed in the
previous section, using your whiteboards and colored markers to draw in the
best fit line: y = -0.6 + 2.9x, which has points (0, -0.6) and (1, 2.3)


<a name="guided-practice"></a>
## Guided Practice: Compute the true best fit line (10 min)

<a name="ind-practice"></a>
## Independent Practice (15 min): Compute the best fit on a final data set

Using the following dataset:

(-1, 0)
(1, 3)
(2, 4)
(3, 7)
(4, 10)

Compute the best fit line using the equations we discussed.

> Have students compute the best fit line and the squared error.
> This data set will have a non-zero squared error.
> The best fit line has coefficients alpha = 1.2972972973, beta = 1.94594594595

> Depending on the amount of time that you have and the class comfort level with
python, you have a few options:
* Have students generate their own data, for example by collecting everyone's
commute time and distance. It doesn't matter if it's a linear relationship, you
can make the quality of fit a discussion point regardless.
* Provide python code to compute the coefficients from the data (below)
* Have students write their own python code.
* Solution code for computing the coefficients and the best fit line is in this
[Jupyter notebook](./code/solution-code/W3-L1-1.ipynb).


<a name="conclusion"></a>
## Conclusion (5 mins)
- Review the concepts of prediction and modeling
- Explain that a linear regression is a model that makes predictions

***


### ADDITIONAL RESOURCES

- [Simple Linear Regression](https://en.wikipedia.org/wiki/Simple_linear_regression)
