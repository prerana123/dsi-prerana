---
title: Study Design
duration: "1:25"
creator:
    name: Marc Harper
    city: LA
---

# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Study Design
Week 3 | Lesson 4.3

### LEARNING OBJECTIVES
*After this lesson, you will be able to:*
- Assess the data you have
- Identify its shortcomings
- Make a clear problem statement / hypothesis
- How do we frame the problem as a potentially valuable business problem
- Outline a high-level plan of action
- How do we communicate the findings

### STUDENT PRE-WORK
*Before this lesson, you should already be able to:*
- Apply critical and scientific thinking

### INSTRUCTOR PREP
*Before this lesson, instructors will need to:*
- Consider adding a personalized scenario of assessing data for your students.
- Generate a brief slide deck
- Prepare any specific materials
- Provide students with additional resources

### LESSON GUIDE
| TIMING  | TYPE  | TOPIC  |
|:-:|---|---|
| 5 min  | [Opening](#opening)  | Discussion  |
| 10 min  | [Introduction](#introduction)   | Study Design  |
| 20 min  | [Guided Practice](#guided-practice<a name="opening"></a>)  | Assessing Data   |
| 15 min  | [Demo](#demo)  | Making a Clear Problem Statement  |
| 20 min  | [Independent Practice](#ind-practice)  | Generating a Hypothesis and Framing it |
| 10 min  | [Demo](#demo)  | Communicating the Findings  |
| 5 min  | [Conclusion](#conclusion)  | Review, Recap  |

---

<a name="opening"></a>
## Opening (5 mins)
- Review prior labs/homework, upcoming projects, or exit tickets, when applicable
- Review lesson objectives
- Discuss real world relevance of these topics
- Relate topics to the [Data Science Workflow](https://drive.google.com/file/d/0Bx2SHQGVqWasOGY4dE95OFVvZjQ/view?usp=sharing) - i.e. are these concepts typically used to acquire, parse, clean, mine, refine, model, present, or deploy?

> This is a good lesson for you to personalize for your students. We'll use a sample dataset and scenario, and you can easily reuse the high level content for another scenario.

> This is also a good time to discuss the Project 3 dataset. Many of these topics will be addressed further in 5.1 and 5.2.

<a name="introduction"></a>
## Introduction: Study Design (10 mins)

Diligent study design will allow you to be more effective and efficient
as a data scientist. In this lesson we will cover principles and procedures
for rigorous study design.

Generally speaking, there are several stages to a study design, including:
- Assessing the data you have
- Identifying its shortcomings
- Making a clear problem statement / hypothesis
- Framing the problem as a potentially valuable business problem or case-study
- Outlining a high-level plan of action
- Communicating the findings

#### Assessing Data

The first step in any study design is planning for the data you need to
investigate your target subject.

* What data do you have?
* What shortcomings does it have? Is it relevant to the chosen problem?
Is it missing a lot of values? Is there a lot of error in measurement?
Is there significant sampling bias?
* Can you obtain other data that will augment the existing data, or act as a
control or validation data set?

Let's take a quick look at the [SF Salary data set](https://www.kaggle.com/kaggle/sf-salaries)
on Kaggle. It contains the following data:

- EmployeeName
- JobTitle
- BasePay
- OvertimePay
- OtherPay
- Benefits
- TotalPay
- TotalPayBenefits
- Year
- Notes
- Agency
- Status

<a name="guided-practice"></a>
## Guided Practice: Assessing Data (20 mins)

> Try a group discussion or Think-Pair-Share here.

Consider this dataset in the following two contexts, identifying strengths
and shortcomings:

- **Salary Growth**: We are commissioned to study the growth of government
employee salaries over the past four years in the state of California.
- **Housing Affordability**: A non-profit has contracted us to determine the affordability of housing in
San Francisco.

> This dataset is pretty good for the **Salary Growth** for SF, but it lacks data
on any other city. Since SF salaries and costs of living have risen quickly,
it may over-estimate salary growth throughout the state, and suffer from bias.

> For the **Housing Affordability** scenario, we would need to obtain housing
cost data and employee residence locations, at least down to the neighborhood level. We also
don't know the marital status of the employees, which may affect housing
affordability.

Another good way to assess data is Exploratory Data Analysis. This will help
you determine if the data is sufficiently complete or has a lot of unusual
or missing values. We'll take a closer look at this data in the next lab.

> You can substitute your own scenario here instead of the project three
data set. Plan to have students discuss in groups or some other We Do
activity.

In the Project 3 data set we have liquor sales data for the state of Iowa down
to the zip code. If we were involved in a health study involving alcohol
consumption rates and the incidences of various diseases and conditions. Assess
the data set in this context.

> **Sample discussion points**: The project 3 data has valuable information on alcohol consumption per capita at the local level, but it has a few shortcomings. Clearly the lack of health data is one. If appropriate health data can be obtained, then aggregate rates of alcohol consumption and disease can be studied, but we're still lacking data on the level of individuals.

Here are some more ways to assess data:

* Verifiability: Can you cross-check specific data points or otherwise verify
accuracy in the data collection process?
* Consider the Source: Does the data come from a reputable organization? Does
the source have any reason to be biased or present misleading data?

No matter how good our study design and analyses are, if you have bad data your
conclusions may be wrong!

<a name="demo"></a>
## Demo/Lecture: Making a Clear Problem Statement (15 mins)

Making a solid hypothesis is crucial to keep a study on track. A good hypothesis
is:

- [Falsifiable](https://en.wikipedia.org/wiki/Falsifiability): can the hypothesis
be proven false?
- [Testable](https://en.wikipedia.org/wiki/Testability) -- Feasibly addressable by the data available or obtainable
- Measurable: the hypothesis can quantified statistically
- Reasonable: An educated guess based on the data
- Specific: Targeted and completable in a specified time frame.

A good hypothesis goes hand-in-hand with the value framing of the problem. Use
these principles to evaluate the following two hypothesis for an A/B test of a
website design. Are the hypotheses testable and falsifiable? Addressable by
the proposed data?

- Our main competitor uses more red in their website. By changing our colors
to be more familiar to their user base, we will encourage customers to
switch to our site.

- By varying the shopping cart navigation widget location in an A/B test and
associating with the final cart total and submission rates, we can determine
if a design change will lead to an increase in revenue.

What's wrong with this hypothesis? It's likely not addressable by the data at
hand. Even if we see an increase in traffic, we would not know if our competitors
traffic decreased. We're making two mistakes -- not framing the hypothesis in
a testable way with the data available, and not being specific enough. This
hypothesis confounds our traffic with our competitors. It's also questionable
if this is an educated guess or not.

### Outlining a Plan of Action

Once we have a good hypothesis, it's time to lay out a plan of action. Recall
the *Data Science Workflow*: Identify, then Acquire, Then Parse, Then Mine.
We need to come up with a plan to:

* Collect/ Obtain the data
* Clean the dataset and prepare to analysis
* Analyze the data,  compute statistics, build models, validate predictions
* Collect and analyze additional data if needed
* Communicate the results

It's important to do a proper cost benefit-analysis and to consider any ethical
implications of a study. If the data is expensive to obtain and the benefits
minimal, your time may be better spent elsewhere. It's also very important
to consider anyone that may be affected (adversely or positively) by the study.
We'll talk about this much more when we get to stakeholder analysis.

<a name="ind-practice"></a>
## Individual Practice: Generating a Hypothesis and Framing it (20 mins)

> You may want to compare/contrast the following with [S.M.A.R.T](https://en.wikipedia.org/wiki/SMART_criteria).

Think of a problem at your workplace (current or past) that you could address
by analyzing data. Follow the Data Science Workflow and the generic outline
above to:

- Describe a dataset (real or fictional)
- Form a hypothesis / write a problem statement
- Layout the value proposition
- Sketch a plan of action.

> Have a few students present their problem statements and give constructive

After discussion, present examples to the rest of the class.

<a name="demo"></a>
## Demo/Lecture: Communicating the Findings (10 mins)

Last but certainly not least is communicating your results. It's critical to
communicate clearly and precisely. Suppose the A/B test we discussed before
showed a small increase in revenue that was not statistically significant.
Brainstorm how you would communicate this result.

> Give students a minute or two to think and write.

Consider the following summary statements and evaluate the accuracy and clarity
of the communication of the results.

- The A/B test of shopping cart location showed a small difference in site
traffic and revenue.
- The A/B test of shopping cart location did show a small difference in site
traffic and revenue, however the change was not statistically significant.
The results of this study do not justify a design alteration.
- The A/B test of shopping cart location did not show a statistically
significant change in site traffic or revenue. The results of this
study do not justify a design alteration.

> The first is confusing and/or misleading. The second frames the positive then
reverses course.

When communicating findings, be sure to:
- State the results clearly
- State the results consistently
- Avoid suggesting unjustified or speculative courses of action.

<a name="conclusion"></a>
## Conclusion (5 mins)

Give a quick rundown of the learning objectives:
- Assessing the data you have
- Identifying its shortcomings
- Making a clear problem statement / hypothesis
- Framing the problem as a potentially valuable business problem or case-study
- Outlining a high-level plan of action
- Communicating the findings

You can also review the definitions of falsifiability, testability, and other
criteria of good hypotheses.

***


### ADDITIONAL RESOURCES

- [Data Science Workflow](http://cacm.acm.org/blogs/blog-cacm/169199-data-science-workflow-overview-and-challenges/fulltext) from ACM
