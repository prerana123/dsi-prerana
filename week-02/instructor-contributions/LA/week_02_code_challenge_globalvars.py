'''
Stuck? Pose this question to yourself: 
	"I could solve this problem, if I know _____________."

1) Create a function that will increment values: 

	assert add(1) == 1
	assert add(2) == 3
	assert add(3) == 6
	assert add(10) == 16

BONUS) How would you change your function to take in 
unknown number of parameters
	
	assert add(2,3,5) == 10
	assert add(2,3,5,1,1,1,1,1,1,1,1,1) == 19


Starting Code: 

	add(n):
		pass

'''

#Solution Part 1:
a = 0

def add(n):
    global a
    a += n
    return a

#Solution Part 2:
#example if used **kwargs then input is a dictionary 
#i.e. dictionary of inputs: {cluster: 1, degrees_freedom: 2}

global_num = 0

def add(*args):
    global global_num
    for arg in args:
        global_num += arg
    return global_num
