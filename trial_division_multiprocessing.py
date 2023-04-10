'''
Program Name: trial_division_multiprocessing.py
Author: Thomas Knoefel
Short Description: This program finds the largest prime number within a specified range using the trial division method and multiprocessing.
Algorithm: Trial Division
Optimization Method: Multiprocessing
Created on: February 20, 2023
Long Description: This program uses the trial division algorithm to find the largest prime number within a range.
'''

import time
from math import isqrt
import multiprocessing

# Function that checks if a given number is prime
def is_prime(n):
    if n <= 1:
        return False
    elif n <= 3:
        return True
    elif n % 2 == 0:
        return False
    else:
        for i in range(3, isqrt(n) + 1, 2):
            if n % i == 0:
                return False
        return True

# Function that counts the number of prime numbers within a range
def count_primes(start, end):
    count = 0
    for i in range(start, end):
        if is_prime(i):
            count += 1
            largest_prime = i
    return count, largest_prime

if __name__ == '__main__':
    start = time.perf_counter()

    varMAX = 100000000
    num_processes = multiprocessing.cpu_count()
    chunk_size = varMAX // num_processes

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Dividing the range into equal parts
        results = [pool.apply_async(count_primes, args=(i, i+chunk_size)) for i in range(3, varMAX, chunk_size)]
        counts = [r.get()[0] for r in results] # Number of prime numbers within each range
        largest_primes = [r.get()[1] for r in results] # Largest prime number within each range

    total_count = sum(counts) # Total number of prime numbers
    largest_prime = max(largest_primes) # Largest prime number in the entire range

    print(f"Largest prime number: {largest_prime}")
    print(f"Number of prime numbers: {total_count}")

    end = time.perf_counter()
    duration = (end - start) * 1000
    print(f"Duration: {duration} ms")
