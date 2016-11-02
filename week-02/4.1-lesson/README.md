---
title: Stats Review & Intro to SciPy
duration: "1:5"
creator:
    name: Lucy Williams
    city: DC
---

# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Stats Review & Intro to SciPy
Week 2 | Lesson 4.1

### LEARNING OBJECTIVES
*After this lesson, you will be able to:*
- Explain what accuracy is and why it can be a flawed metric
- Explain Type I and Type II errors
- Explain t-testing
- Demonstrate t-testing using scipy

### INSTRUCTOR PREP
*Before this lesson, instructors will need to:*
- Read in / Review any dataset(s) & starter/solution code
- Generate a brief slide deck

### LESSON GUIDE
| TIMING  | TYPE  | TOPIC  |
|:-:|---|---|
| 5 min  | [Introduction](#introduction)   | Accuracy, Type I and Type II errors, t-testing  |
| 10 min  | [Demo /Guided Practice](#demo)  | Accuracy  |
| 10 min  | [Demo /Guided Practice](#demo)  | Type I and Type II errors  |
| 40 min  | [Demo /Guided Practice](#demo)  | t-testing  |
| 20 min  | [Independent Practice](#ind-practice)  |   |
| 5 min  | [Conclusion](#conclusion)  | |

---

<a name="Accuracy, Type I and Type II errors, t-testing"></a>
## Introduction: Accuracy, Type I and Type II errors, t-testing (5 mins)

- Accuracy refers to how close a sample statistic is to a population parameter .
- Type I error. A Type I error occurs when the researcher rejects a null hypothesis
    when it is true.
- Type II error. A Type II error occurs when the researcher accepts a null hypothesis
    that is false.
- A t-test is any hypothesis test in which the test statistic follows
    Student's t distribution if the null hypothesis is true. We'll discuss the
    Student's t distribution and some common t-tests below.

**Check:** Why do you think that accuracy, type I and type II errors, and t-testing are important?

- [accuracy](http://stattrek.com/statistics/dictionary.aspx?definition=Accuracy)
- [type errors](http://stattrek.com/statistics/dictionary.aspx?definition=type errors)
- [t-test](http://stattrek.com/statistics/dictionary.aspx?definition=t-test)


<a name="Accuracy"></a>
## Demo/Guided Practice: Accuracy (10 mins)

Accuracy refers to the closeness of a measured value to a standard or known value.
It is often confused with precision, but precision is the closeness of two or more
measurements to each other.  

A good analogy for understanding accuracy and precision is to imagine a basketball
player shooting baskets. If the player shoots with accuracy, his aim will always
take the ball close to or into the basket. If the player shoots with precision, his
aim will always take the ball to the same location which may or may not be close to
the basket. A good player will be both accurate and precise by shooting the ball the
same way each time and each time making it in the basket.

**Check:** Is it better to be accurate of precise? Why?
- [accuracy](https://www.ncsu.edu/labwrite/Experimental%20Design/accuracyprecision.htm)



<a name="Type I and Type II errors"></a>
## Demo/Guided Practice: Type I and Type II errors (10 mins)

Remember we defined a Type I error as an error that occurs when the researcher
rejects a null hypothesis when it is true. The probability of committing a Type I
error is called the significance level , and is often denoted by α.

Remember we defined a Type II error as an error that occurs when the researcher
accepts a null hypothesis that is false.  The probability of committing a Type II
error is called Beta, and is often denoted by β.

This table summarizes Type I and Type II errors.

![](https://github.com/generalassembly-studio/dsi-course-materials/blob/W2-L4.1/curriculum/04-lessons/week-02/4.1-lesson/assets/images/errorTable.png)

**Check:** Why are Type I and Type II errors important?
- [type I and type II errors](http://stattrek.com/statistics/dictionary.aspx?definition=Accuracy)
- [error table ](https://www.ma.utexas.edu/users/mks/statmistakes/errortypes.html)


<a name="t-testing"></a>
## Demo/Guided Practice: t-testing (40 mins)

Remember a t-test is any hypothesis test in which the test statistic follows
Student's t distribution if the null hypothesis is true. But first, let's
talk about the Student's t distribution.

The Student's t distribution is a probability distribution that is used to
estimate population parameters when the sample size is small and/or when the
population variance is unknown.  According to the central limit theorem, the
sampling distribution of a statistic (like a sample mean) will follow a normal
distribution, as long as the sample size is sufficiently large. But sample sizes
are sometimes small, and often we do not know the standard deviation of the
population. When either of these problems occur, statisticians rely on the
distribution of the t statistic. The t distribution allows us to conduct
statistical analyses on certain data sets that are not appropriate for analysis,
using the normal distribution. The t distribution can be used with any statistic
having a bell-shaped distribution (i.e., approximately normal)

A few common t-tests include:
- One-sample t-test. Used to determine whether a hypothesized population
    mean differs significantly from an observed sample mean.
- Two-sample t-test. Used to determine whether the difference between samples means differs significantly from the              hypothesized difference between population means.
- Paired t-test. Used to test the significance of the difference
    between paired means.

Let's take a look at each of these types of t-tests.

The 1-sample t-test is used when we want to compare a sample mean to a population
mean (which we already know). The average British man is 175.3 cm tall. A survey
recorded the heights of 10 UK men and we want to know whether the mean of the sample
is different from the population mean.

```Python
import numpy as np
import scipy as sp
from scipy import stats
```

Input the one sample data:

```Python
one_sample_data = [177.3, 182.7, 169.6, 176.3, 180.3, 179.4, 178.5, 177.2, 181.8, 176.5]
```

Find the t-statistic and the p-value:

```Python
one_sample = stats.ttest_1samp(one_sample_data, 175.3)
print "The t-statistic is %.3f and the p-value is %.3f." % one_sample
```
We can conclude that the average height of our sample is significantly different
(p <  0.05) from the average British male height. The return value is the result of
a two-sided t-test and is a tuple containing the t-value and the p-value.


**Check:** Why are t-tests useful?
- [t-test](http://stattrek.com/statistics/dictionary.aspx?definition=t-test)
- [t distribution](http://stattrek.com/probability-distributions/t-distribution.aspx)
- [t-tests](http://iaingallagher.tumblr.com/post/50980987285/t-tests-in-python)


<a name="ind-practice"></a>
## Independent Practice: Topic (20 minutes)



**Check:** Were students able to create the desired deliverable(s)? Did it meet requirements / constraints?



<a name="conclusion"></a>
## Conclusion (5 mins)

> Review concepts covered:
- Explain what accuracy is and why it can be a flawed metric
- Explain Type I and Type II errors
- Explain t-testing
