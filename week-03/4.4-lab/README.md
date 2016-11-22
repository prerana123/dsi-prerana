---
title: Case Study
type: lab
duration: "1:25"
creator:
    name: Marc Harper
    city: LA
---

# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Case Study Lab

## Introduction

You are a data scientist at a top recruiting firm. Predict job salary based on different attributes to help the company more accurately/profitably determine compensation.

Take extra care to apply the learning objectives from 4.3:
- Assess the data you have
- Identify its shortcomings
- Make a clear problem statement / hypothesis
- How do we frame the problem as a potentially valuable business problem
- Outline a high-level plan of action
- How do we communicate the findings

Since we only have so much time, try to spend more time applying and discussing
these objectives rather than finding a perfect model. You may want to restrict
your attention to a single job type and further refine the scenario.

> There are a number of similar projects like this on the web for this dataset,
some on the Kaggle website. It's much better to focus on the qualitative aspects.

> Some suggestions:
* Have students focus on a single job type or a small number of jobs
* Have students work in small groups where each focuses on a single job in the
same industry, E.g. medical, public service, police

> The [solution code](./code/solution-code/) simply computes the median salary of all firefighters for
2011-2014, fits a linear model, and extrapolates to 2015. This should be enough
content to scaffold a narrative around and practice the softer skills of this
lesson and the previous one.

> At this point students should be pretty good at fitting linear models. This is
the time to really work on the other aspects such as clear presentation.

#### Requirements

- Read in the data
- Perform a linear regression
- Visualize the results
- Communicate the results clearly

**Bonus:**
- Take a look at Total Pay + Benefits as well. Benefits are an important part of
compensation offers and your stakeholder may be interested.
- Compare growth rates across positions and check if that aligns with your
earlier prediction(s)
- Identify any significant changes in the number of positions held per job title
Is any particular job title growing or decline? That could affect growth rates
in salary.

#### Starter code

There is a loose outline in the [starter code](./code/starter-code/4.4-Case-Study-Starter.ipynb) to get you to a model and a prediction. The write-up around that is up to you. Pay close attention to the principles
above.

> [Solution Code](./code/solution-code/4.4-Case-Study-Solution.ipynb)

#### Deliverable

A jupyter notebook with your write-up, including a visualization of your prediction.

## Additional Resources

- There are some [writeups for this dataset](https://www.kaggle.com/kaggle/sf-salaries)
on Kaggle. If you get stuck take a look, but give it your best shot first.
