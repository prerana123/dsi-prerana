'''
Week 3 Tuesday Code Challenge 

You are given the following information, 
but you may prefer to do some research for yourself.

    1 Jan 1900 was a Monday.
    Thirty days has September,
    April, June and November.
    All the rest have thirty-one,
    Saving February alone,
    Which has twenty-eight, rain or shine.
    And on leap years, twenty-nine.
    A leap year occurs on any year evenly divisible by 4, but not on a century unless it is divisible by 400.

How many Sundays fell on the first of the month 
during the twentieth century (1 Jan 1901 to 31 Dec 2000)?

Starter Code:

import pandas as pd
from datetime import timedelta, date

def generate_first_sundays(start_date, end_date):
    a = pd.date_range()
    pass

start_date = date(1901, 1, 1)
end_date = date(2000, 12, 31)

assert(generate_first_sundays(start_date, end_date)) == 171

'''

#METHOD 1 - THANKS MIKE!
def generate_first_sundays(start_date, end_date):
    i = pd.date_range(start_date,end_date,freq = 'w',offset='W-SUN')
    return len([j for j in i if j.day == 1])


#METHOD 2
def generate_first_sundays(start_date, end_date):
    '''
    parameter dt_rng: date range with start and end dates
    returns count of the number of First sundays within the date range

    '''
    rng = pd.date_range(start=start_date, end=end_date, freq='D')

    count = 0

    for date in rng:
        if date.day == 1 and date.weekday() == 6:
            count += 1
    return count

start_date = date(1901, 1, 1)
end_date = date(2000, 12, 31)
generate_first_sundays(start_date, end_date)


#METHOD 3
from datetime import timedelta, date

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def generate_first_sundays(start_date, end_date):
    answer_days = 0

    for single_date in daterange(start_date, end_date):
        
        if (single_date.weekday()== 6) & (single_date.day == 1):
            
            print(single_date.strftime("%Y-%m-%d") + ' is a Sunday!')
            
            answer_days += 1
    
    return answer_days

start_date = date(1901, 1, 1)
end_date = date(2000, 12, 31)
generate_first_sundays(start_date, end_date)

