'''
Your task is to write function which takes string and 
list of delimiters as an input and returns list of strings/characters 
after splitting given string.

Output 1: 
assert(multiple_split('Hi, how are you?', [' ']) ) == [Hi,', 'how', 'are', 'you?']
assert(multiple_split('1+2-3', ['+', '-'])) == ['1', '2', '3']

If you get the above output to work, also try to run this: 
assert(multiple_split('somestring')) == ['somestring']

Starter Code: 
*notice, function parameter has default parameters

def multiple_split(string, delimiters=[]):
    "Your function goes here"

'''


#Method 1  
def multiple_split(string, delimiters=[]):
    arr = [string]

    for d in delimiters:
        for i in range(len(arr)-1,-1,-1):
            arr[i:i+1] = arr[i].split(d)
    
    return list(filter(lambda x: x != '', arr))



#METHOD 2 - Change all delimiters to one word, then split on that word

def multiple_split(text, delims=[]):
    if not delims:
        return [text] if text else []
    
    for delim in delims:
        str = text.replace(delim, 'DELIM')
        text = str

    text = text.split('DELIM')
    return [word for word in text if word != '']

#METHOD 3 - clever use of 're' library + python's filter and map

from re import split, escape

def multiple_split(string, delimiters=[]):
    
    return filter(None, split('|'.join(map(escape, delimiters)), string))








#METHOD -- only works for first 2 outputs, does not work for one simple string
from itertools import chain

def multiple_splits(string, delimiters=[]):

    if delimiters:
        for d in delimiters: 

            if isinstance(string, list):
                string = [s.split(d) for s in string]
                string = list(chain.from_iterable(string))
            
            else: 
                string = string.split(d)

    return list(filter(None, string))

#METHOD 1 
def multiple_split(string, delimiters=[]):

    for d in delimiters: 
        if isinstance(string, list) and d:
            string = [filter(None, s.split(d)) for s in string]
        else: 
            string = string.split(d)

    return [o for out in string for o in out]