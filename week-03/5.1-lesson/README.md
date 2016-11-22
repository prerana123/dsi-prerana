---
title: Stakeholder Analysis
duration: "1:25"
creator:
    name: Marc Harper
    city: LA
---

# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Stakeholder Analysis
Week 3 | Lesson 5.1

### LEARNING OBJECTIVES
*After this lesson, you will be able to:*
- How to identify different stakeholders
- What are their incentives/ what do they care about
- How to frame your analysis and results in context of what matters to them

### STUDENT PRE-WORK
*Before this lesson, you should already be able to:*
- Compute basic statistics
- Fit linear models to data
- Compute regression metrics and model fits

### INSTRUCTOR PREP
*Before this lesson, instructors will need to:*
- Brainstorm 1-2 good examples of audiences you have presented to in the past
and be ready to discuss your experience in terms of stakeholder identification
and analysis.
- Read in / Review any dataset(s) & starter/solution code
- Generate a brief slide deck
- Prepare any specific materials
- Provide students with additional resources

### LESSON GUIDE
| TIMING  | TYPE  | TOPIC  |
|:-:|---|---|
| 5 min  | [Opening](#opening)  | Discussion  |
| 15 min  | [Introduction](#introduction)   | Identifying Stakeholders |
| 10 min  | [Demo](#demo)  | Sample Stakeholder Identification  |
| 15 min  | [Guided Practice](#guided-practice<a name="opening"></a>)  | Stakeholder Incentives |
| 10 min  | [Demo](#demo)  | Framing Your Analysis and Results |
| 10 min  | [Guided Practice](#guided-practice<a name="opening"></a>)  | Good and Bad Visualizations   |
| 15 min  | [Independent Practice](#ind-practice)  | Project 3 Stakeholders |
| 5 min  | [Conclusion](#conclusion)  | Review / Recap  |

---

<a name="opening"></a>
## Opening (5 mins)
- Review prior labs/homework, upcoming projects, or exit tickets, when applicable
- Review lesson objectives
- Discuss real world relevance of these topics
- Relate topics to the [Data Science Workflow](https://drive.google.com/file/d/0Bx2SHQGVqWasOGY4dE95OFVvZjQ/view?usp=sharing) - i.e. are these concepts typically used to acquire, parse, clean, mine, refine, model, present, or deploy?

> **Check:** Review students' progress on Project 3. Which students chose each scenario? How are they thinking about framing their results at this point?

> This lesson will be much less technical than most of the other lessons this week.

<a name="introduction"></a>
## Introduction: Identifying Stakeholders (15 mins)

One of the most important aspects of data science is communication, particularly
when stakeholders are involved. Results must be communicated effectively to
clients, customers, and managers, taking their needs and technical backgrounds
into account.

When communicating with fellow data scientists and other technical folks we
naturally use a lot of technical jargon. This is usually not appropriate for
other audiences.

_Primary stakeholders_ include anyone that your analysis will be presented to
and whom will act on your conclusions and recommendations as well as
_secondary stakeholders_, anyone significantly affected by your efforts.
Stakeholders will sometimes have technical backgrounds but typically this is
not the case.

Identifying stakeholders is the first step in determining the right tone and
level of technical depth to use when communicating analysis and results. Often
the primary stakeholder is the party that hired or contracted you, and there
may be many stakeholders in any given scenario.

A good first approximation at identifying primary stakeholders is by their
influence. Executives and other leaders, large shareholders, government
officials, and people in positions of influence such as college presidents,
department heads, project managers, and community leaders are all good examples.
Secondary stakeholders tend to be those that are influenced -- rank and file
employees, community members, students and others that are more often affected
by the decisions of those with influence.

### Exercise

**Identify the stakeholders**
* An ad-tech company has hired you to evaluate the accuracy of their click-rate
metrics and analyze their traffic for fraudulent clicks. Specifically you were
hired by an engineering manager that believes that the metrics are inaccurate.
* You are hired by an engineering consulting firm to analyze data from a
recent industrial accident. The consulting firm was hired by a factory owner
to determine the root cause of an explosion, and the owner is considering
litigation over a faulty component.
* You are hired by a local librarian to model the likelihood of borrowed books
being returned based on various demographics and historical data.

> Some Stakeholders:
* The company and the manager that hired you; the project manager of the
ad-network's analytics, the companies that purchase the ads, other employees
* The engineering consulting firm, the factory owner, the component manufacturer
* The local library and its patrons (eventually policy changes will have to
be communicated and justified to the borrowers), possibly the city manager or
other local government official that directs library funding

<a name="demo"></a>
## Demo: Sample Stakeholder Identification (5-10 mins)

> Present a demo stakeholder identification (or more than one) relevant to
the background of your students. If possible, identify their motivations as a
hook for the next section.

<a name="guided-practice"></a>
## Guided Practice: Stakeholder Incentives (20-30 mins)

Once you've identified stakeholders, the next step in stakeholder analysis
is to understand their interests, motivations, and incentives.

> This is a good group-discussion activity.

Let's discuss the stakeholders and their incentives in the above scenarios. In
each case you've been hired to discover something from a set of data. Often
a stakeholder will have an expectation of the results. It's crucial to
* Manage expectations throughout
* Present data-based analysis with evidence-backed explanations
* Communicate results with an authoritative and non-confrontational tone

For each of the scenarios above, try to identify the stakeholders' motivations
and expectations. If you are not sure, try to think of some plausible motivations.

> You can have students theorize potential motivations or present some ideas
first.

To do these effectively you need to understand the stakeholder's motivations
and incentives. There are many possibilities. The stakeholder may simply:
- Need help analyzing data
- Have a point to prove or disprove
- Have a desire for technical justification for a hypothesis

These high-level incentives often have a more direct incentive:
- Increasing profit -- cutting costs, optimizing pricing, identifying production
bottlenecks
- Social change -- the stakeholders may be trying to justify an inequality or
quantify its effects
- Many others -- can you think of any?


## Demo: Framing Your Analysis and Results (5-10 mins)

> Back to slides or lecturing

As a data scientist you will have significant influence over the decisions
of stakeholders and institutions. It is crucial that you remain objective
and unbiased with your analyses and conclusions. Often we are tempted to
please or agree with those that we work with or are contracted to. It is a
disservice to all the other stakeholders to slant your conclusions to suit
the desires of one stakeholder over another.

Even more crucially, you must take care to communicate your results in direct and understandable language, including enough technical information to justify
your assertions. It's fine to explain your results in the context of your
stakeholders' interests so long as you can maintain objectivity.

Good visualizations are critical to the communication of results. Avoid
excessive details. Plots are better than tables. That said, avoid unnecessary
visualizations -- each plot should make a concrete point clearly. Keep your
plots simple -- two simple plots are better than one complex plot. Just as we must balance bias and variance in our models, we must also balance complexity
and information in our figures.

It's tempting to make complex figures -- when they land it can be quite
dramatic:

![](http://blog.claricetechnologies.com/wp-content/uploads/2012/09/1.Minard.png)
But for every good complex visualization there are at least ten bad attempts.

<a name="guided-practice"></a>
## Guided Practice: Good and Bad Visualizations (20 mins)

> Take a look at examples of good and bad visualizations with the class or in
small groups

> [Examples](http://gizmodo.com/8-horrible-data-visualizations-that-make-no-sense-1228022038) | [There are entire websites based on showcasing bad visualizations](http://viz.wtf/)

>Have students practice different communication techniques to different
stakeholders. For example, in the library scenario above
the goal is to, sketch out a few visualizations of results that you might
communicate to the librarian and government employees, and another to the
library patrons (perhaps a flyer informing them of new policies and why).

<a name="ind-practice"></a>
## Independent Practice: Project 3 Stakeholders (15 minutes)

> If you are short on time, students will pick this up in the next lab. Go with
the class dynamic -- if the group discussions have been going well, continue
with them.

Spend some time practicing what we've learned today for your Project 3 scenario.
Your tasks are to:

* Identify stakeholders, primary and secondary
* Identify stakeholder interests, motivations, and incentives
* Sketch (on paper) some potential visualizations of your anticipated findings
to various stakeholders

<a name="conclusion"></a>
## Conclusion (5 mins)
- Recap topic(s) covered in short takeaway bullet points, such as:
(a) importance of identifying stakeholders
(b) importance of proper framing of analysis and results
(c) what makes a good visualization.
- Describe homework or any upcoming tasks

***


### ADDITIONAL RESOURCES

- [Five Questions to Identify Stakeholders](https://hbr.org/2014/03/five-questions-to-identify-key-stakeholders/)
- [Identifying Stakeholders](http://ctb.ku.edu/en/table-of-contents/participation/encouraging-involvement/identify-stakeholders/main)
- [Nice article for community work stakeholder analysis](http://ctb.ku.edu/en/table-of-contents/participation/encouraging-involvement/identify-stakeholders/main)
- [Data Visualization](http://tech.globant.com/en/data-and-information-visualization/), aesthetics, and complexity
