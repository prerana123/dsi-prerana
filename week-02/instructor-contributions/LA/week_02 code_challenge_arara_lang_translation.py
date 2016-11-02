'''
Code Challenge:

The Arara are an isolated tribe found in the Amazon who count in pairs. 
For example one to eight is as follows:
https://en.wikipedia.org/wiki/Arara_people

1 = anane 
2 = adak 
3 = adak anane 
4 = adak adak 
5 = adak adak anane 
6 = adak adak adak
7 = adak adak adak anane
8 = adak adak adak adak 

Take a given number and return the Arara's 
equivalent

# optional - Use pytest package to enable testing 

assert count_arara(3) == 'adak anane'
assert count_arara(8) == 'adak adak adak adak'

'''



### METHOD 1 -- "".join()
def count_arara(n):
    return ' '.join(['adak'] * (n / 2) + ['anane'] * (n % 2))


### METHOD 2 -- str.rstrip + .join
def count_arara(n):
    return str.rstrip(''.join(["adak " * (n / 2), "anane" * (n % 2)]))

## METHOD 3 - - Thanks Sharam and Michael Gat!
def Arara(n):   
    return ('adak '*(n/2) if (n%2==0) else 'adak '*(n/2)+'anane ')[:-1]

### METHOD -- not reviewed in class, happy googling
from itertools import chain, repeat

def count_arara(n):
    twos = repeat('adak', n / 2)
    one = repeat('anane', n % 2)
    return ' '.join(chain(twos, one))

## METHOD 1 -- divmod strip
def count_arara(n):
    a,b = divmod(n, 2)
    s = "adak " * a + "anane" * b
    return s.strip()

### METHOD 2 -- divmod + .join + list comprehension
def count_arara(n):
    adaks, anane = divmod(n, 2)

    elements = ["adak" for i in xrange(adaks)]
    if anane:
        elements.append("anane")

    return " ".join(elements)
