'''
Week 3 Code Challenge - Wednesday

Implement a function is_vowel(str), which should return true if a given object is a vowel, false otherwise.
Use regex.

Try two solutions: 
1) use the package re
2) use pandas 

starter code: 

	is_vowel(str):
	    pass
    

'''

#METHOD 1 - THANKS BRETT!

import re
def is_vowelREG(string):
    VFinder = re.compile(r'[aeiou]|[AEIOU]')
    if VFinder.findall(string):#.lower()):
        return True
    else:
        return False
​
is_vowelREG('E')

#METHOD 2
import re

def is_vowel(word):

    if re.match("[^aeiou]*[aeiou]+$",word.lower()):
        print "TRUE"
    else:
        print "FALSE"
	
word = raw_input('Enter a word:')
is_vowel('e')


