---
title: Lambda Functions & Missing Data
type: lab
duration: "1:5"
creator:
    name: Lucy Williams
    city: DC
---

# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Lambda Functions & Missing Data

### Lambda

Lambda is a tool for building functions. We already know how to build functions
using `def`, but let's do a quick comparison of the two.


Here's building a function using `def`:
```Python
def square_root(x): return math.sqrt(x)
```

Here's building the same function using `lambda`:
```Python
square_root_lambda = lambda x: math.sqrt(x)
```

We know that functions are usually created to reduce code duplication or to modularize code. But suppose you need to create a function that is going to be used only once — called from only one place in your application. What would you do then?

Well, first of all, you *don’t need to give the function a name*. It can be “anonymous”. And you can just define it right in the place where you want to use it. That’s where lambda is useful.

Some things to remember about lambda:
- it can only take a single expression
- it does not contain a return statement
- it is a tool for creating anonymous procedures

More information on [Lambda](https://pythonconquerstheuniverse.wordpress.com/2011/08/29/lambda_tutorial/).


### Missing data

Missing data is also known as `NA` values.  By “missing” we simply mean null or “not present for whatever reason”. Many data sets simply arrive with missing data, either because it exists and was not collected or it never existed.

Use the `isnull()` method to detect the missing values. The output shows `True` when the value is missing. By adding an index into the dataset, you obtain just the entries that are missing.

```Python
import pandas as pd
import numpy as np
s = pd.Series([1, 2, 3, np.NaN, 5, 6, None])
print s.isnull()
print
print s[s.isnull()]
```
A dataset could represent missing data in several ways. In this example, you see missing data represented as `np.NaN` which stands for "NumPy Not a Number" and the Python `None` value.


### Fill in missing data

To fill in missing data use `fillna()`. For `fillna()` you need to provide a number. Usually, the mean, median, or mode is used. Let's use the same data set and this time let's fill in missing values with the mean.

```Python
s = pd.Series([1, 2, 3, np.NaN, 5, 6, None])
print s.fillna(int(s.mean()))
```

We could also just drop all the NAs, by using `dropna()`:
```Python
print s.dropna()
```

Note: Here is some information on dealing with [missing data](http://pandas.pydata.org/pandas-docs/stable/missing_data.html).


#### Requirements

Write the equivalent `lambda` function for the following `def` function:

```Python
def f (x): return x**2
print f(8)
```

Use the following [Crunchbase dataset](https://raw.githubusercontent.com/suneel0101/lesson-plan/master/crunchbase_monthly_export.csv) and:

- Detect missing data
- Fill in missing data with the mean

### Starter Code

[Starter code can be found here](./code/starter-code/w2-3.4-starter.ipynb).

> [Solution code](./code/solution-code/w2-3.4-solution.ipynb).
