---
title: Prepare Visuals for Project 4
type: lab
duration: "1:25"
creator:
    name: Jonathan Balaban
    city:
---

# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 4 Visualization

## Introduction

> ***Note:*** _This can be a pair programming activity or done independently._

Drawing from best practices covered in our lesson (5.1), use Tableau to generate a visualization of your weekly project and share results with the class.

#### Requirements

- Create clear communication and approachable visual representations
- Communicate  main insights from your data using classification
- Use graphical displays
- Proactively highlight strengths and weaknesses of your data, model, predictions, and recommendations

#### Guidelines

- Use labels, titles, and legends for your plots
- Provide context around your project with a clear problem statement, modeling strategy, and hypothesis
- List your model and data assumptions, strengths, and weaknesses

Here's an example of labeling and titling in python:

```python
plt.plot(jobs['Data'])

plt.title('Title of my Plot')
plt.xlabel('Data Science; 1=Yes, 0=No')
plt.ylabel('Age')
```
In Tableau, build a histogram and at least one other visualization on your scraped USAJobs.com data.

> ***Hint:*** To bin continuous variables in Tableau, go to Data > Create > Bins

Remember to ask:

- Who: Who is my target audience for this visual?
- What: What do they already know about this project? What do they need to know?
- How: How does my project affect this audience? How might they interpret (or misinterpret) the data?

#### Deliverable

- Visualize different aspects of your data and machine learning model (accuracy plots, feature correlations, etc.)
- Quickly share your visualizations, focusing on actionable information (2-3 mins)
- Highlight key facts that relate to the your project data

## Additional Resources
- [5 best practices for telling great stories with data](https://drive.google.com/file/d/0Bx2SHQGVqWasTmhYM1FHX3JfNEU/view)
- [Learn by example: Tableau visualizations](http://www.tableau.com/blog/learning-example-real-data-visualizations)
