'''
Program Name: trial_division.py
Author: Thomas Knoefel
Short Description: This program finds the largest prime number within a specified range using the trial division method.
Algorithm: Trial Division
Optimization Method: None
Created on: February 20, 2023
Long Description: This program uses the "Trial Division" method to find the largest prime number within a specified range.
'''

import time
from math import isqrt


start = time.time()
start_proc = time.process_time()

varMAX=100000000
count=1

for i in range(3,varMAX+1,2):
    prime_found = True
    for j in range(3,isqrt(i)+1,2):
        remainder= i % j
        if remainder == 0:
            prime_found = False
            break
    if prime_found:
        largest_prime=i
        count += 1

print("Largest prime number:", largest_prime)
print("Number of prime numbers:", count)

end = time.time()
end_proc = time.process_time()
print('Total time: {:5.3f}s'.format(end-start))
print('System time: {:5.3f}s'.format(end_proc-start_proc))
