'''
Implement the function unique_in_order which 
takes as argument a sequence and returns a list 
of items without any elements with the same 
value next to each other and preserving the 
original order of elements.

For example:

unique_in_order('AAAABBBCCDAABBB') == ['A', 'B', 'C', 'D', 'A', 'B']
unique_in_order('ABBCcAD')         == ['A', 'B', 'C', 'c', 'A', 'D']
unique_in_order([1,2,2,3,3])       == [1,2,3]
'''






















#METHOD 1 CODE 

def unique_in_order(iterable):
    list_iterable = list(iterable)         
    list_new = []
    for i in range(0, len(list_iterable)):
        if i == 0:
            check = list_iterable[i]
            list_new.append(list_iterable[i])
        elif i > 0: 
            if check == list_iterable[i]:
                pass
            else: 
                check = list_iterable[i]
                list_new.append(list_iterable[i])
    return list_new

#METHOD 2 CODE

def unique_in_order(iterable):
    result = []
    prev = None
    for char in iterable[0:]:
        if char != prev:
            result.append(char)
            prev = char
    return result

#METHOD 3 CODE
#Review class groupby(object) in docs: 
# https://docs.python.org/2/library/itertools.html#itertools.groupby
from itertools import groupby

def unique_in_order(iterable):
    return [k for (k, _) in groupby(iterable)]

import pytest

testCase.assert_equals(unique_in_order('AAAABBBCCDAABBB'), ['A','B','C','D','A','B'])


# ADDIITONAL CODE FOR RANDOM SERIES CREATION
#create a Series with index == datetime
rng = pd.date_range('1/1/2015', periods=72, freq='')
ts = pd.Series(np.random.randn(len(rng)), index=rng)

#optional fun functions on timeseries
converted = ts.asfreq('45Min', method='pad')

ts.resample('D').mean()

            