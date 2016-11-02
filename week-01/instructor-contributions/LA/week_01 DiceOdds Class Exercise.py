'''
Question:
There are 36 possible combinations of two dice. 
A simple pair of loops over range(6)+1 will enumerate all combinations. 
The sum of the two dice is more interesting than the actual combination. 
Create a dict of all combinations, using the sum of the two dice as the key.

Each value in the dict should be a list of tuples; each tuple has the value of two dice. 
The general outline is something like the following:

Run the code in jupyter notebook or ipython in terminal (hint: "%paste" will copy
	your clipboard directly into the terminal in ipython)

PSEUDO CODE:
dice_dict = {}

Loop with d1 from 1 to 6

    Loop with d2 from 1 to 6
        newTuple ← ( d1, d2 ) # create the tuple
        oldList ← dictionary entry for sum d1+d2
        newList ← oldList + newTuple
        replace entry in dictionary with newList

Loop over all values in the dictionary
    print the key and the length of the list

SAMPLE OUTPUT 
	#key: value == {1: [(0,1), (1,0)]}

A. Create the dictionary
B. Print the dictionary: values, keys, values and keys
C. BONUS: With the dictionary that you created (dice_dict), 
create a new dictionary of dice combination and probabilities associated with each of these combinations. 
The new dictionary should have have key: value == {1: 0.0556}
'''

dice_dict = {} #create empty dictionary

for d1 in range(1,7):
	for d2 in range(1,7):
		comboTuple = ( d1, d2 ) # create the tuple
		key_sum = d1 + d2
		if key_sum is not in dice_dict.keys():
			dice_dict[key_sum] = comboTuple
		else: 
			dice_dict[key_sum].append(comboTuple)


# SECTION A. CREATE THE DICTIONARY
#1. Creating dictionary you may have encountered a "KEY ERROR"
# Except to get an error for 
dice_dict = {} #initialize 
for d1 in range(1,7): #loops over value of first dice
	for d2 in range (1,7): #loops over value of second dice
		newTuple = (d1, d2)
		key_sum = d1 + d2 
		dice_dict[key_sum].append(newTuple) #KEY ERROR for not initializing dict

#2. First elements, initializing variables are important
# Even dictionary and lists to create the final dictionary

dice_dict = {} #initialize 

for d1 in range(1, 7): #loops over value of first dice
	for d2 in range (1, 7): #loops over value of second dice
		newTuple = (d1, d2)
		key_sum = d1 + d2 
		if not key_sum in dice_dict: 
			dice_dict[key_sum] = [(newTuple)]
		else: 
			dice_dict[key_sum].append(newTuple)


'''
SECTION B. After creating the dictionary, print out elements of the dictionary
'''

#print the dictionary keys in a loop
for keys in dice_dict.keys():
	print keys

#print the dictionary values in a loop
for value in dice_dict.values():
	print value

#print the dictionary values and in a loop
for key,value in dice_dict.iteritems():
	print key,":",value

