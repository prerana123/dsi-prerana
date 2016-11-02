'''
Wednesday Morning Challenge

Question: 
The numberOfOccurrences function must return the 
number of 
occurrences of an element in an array.

Example: 
sample = [0,1,2,2,3]
xs = 0

assert 1 == 1
assert number_of_occurrences([0,1,2,2,3], 0) == 1
assert number_of_occurrences([0,1,2,2,3], 4) == 0
assert number_of_occurrences([0,1,2,2,3], 2) == 2
assert number_of_occurrences([0,1,2,2,3], 3) == 1

Starter code:

def number_of_occurrences(sample, xs)
	pass 

'''











#Method 1
def number_of_occurrences(s, xs):
    total = 0
    for x in xs:
        if x == s:
            total+=1
    return total







#Method 2
def number_of_occurrences(s, xs):
    return s.count(xs)









#Method 3

def number_of_occurrences(s, xs):
    return sum(xs == y for y in s)












#Method 4 - helper package collections

def number_of_occurrences(s, xs):
    from collections import Counter

    print Counter(sample)[xs]

#Method 5 - lambda functions
def number_of_occurrences(sample, xs):
    equalsS = lambda x: x == xs  
    return len(filter(equalsS ,sample))

# This is equivalent 
def number_of_occurrences(sample, xs): 
    tmp = [item for item in sample if item == xs] 
    return len(tmp)

# If time introduction assertions
# conda install or pip install pytest
import pytest 

assert 











