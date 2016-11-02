'''
Extention to the dice odds

C. BONUS: With the dictionary that you created (dice_dict), 
create a new dictionary of dice combination and probabilities associated with each of these combinations. 
The new dictionary should have have key: value == {1: 0.0556}

**Remember to run dice_dict from first part of this exercise

PSEUDO CODE:
- initialize dice_prob dictionary
- Count the number of total combinations (list comprehension)
 - Loop through the dictionary and divide total results 
	for a result with total combinations

'''
dice_prob = {}

list_count = [len(dice_dict[d]) for d in dice_dict]
total_count = sum(list_count)

result_count = 0
for d in dice_dict: 
	for i in dice_dict[d]:
		result_count += 1
	dice_prob[d] = float(result_count) / float(total_count)
	result_count = 0


## BONUS: HOW TO RETURN KEY VALUE ASSOCIATED WITH MAXIMUM VALUE IN THE DICTIONARY

#use max function in python
max(dice_prob, key=lambda x: dice_prob[x])

#use collections package - Counter and most_common
import collections

tmp = collections.Counter(dice_prob)
tmp.most_common(1)




