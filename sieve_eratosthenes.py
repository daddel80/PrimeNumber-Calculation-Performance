'''
Program Name: sieve_eratosthenes_numpy.py
Author: Thomas Knoefel
Short Description: This program calculates the number of prime numbers and the largest prime factor of n.
Algorithm: Sieve of Eratosthenes
Optimization Method: None
Created on: February 20, 2023
Long Description: This program uses the Sieve of Eratosthenes to find all prime numbers up to a given value n.
'''

import time
import numpy as np

# Definition of the function to determine the number of prime numbers
def count_primes(n):
    # Creating a NumPy array that assumes all numbers up to n are prime
    is_prime = np.ones(n, dtype=bool)
    is_prime[0:2] = False  # Setting the first two elements of the array to False (0 and 1 are not prime numbers)
    num_primes = 0
    for i in range(2, n):
        if is_prime[i]:
            num_primes += 1
            # Marking all multiples of the current number as not prime
            is_prime[i*i::i] = False
    # Returning the number of prime numbers and the largest prime factor of n
    return num_primes, np.max(np.nonzero(is_prime)[0])

if __name__ == "__main__":
    begin = time.perf_counter()  # Starting time measurement
    n = 10000000000  # Determining the upper limit n for the search for prime numbers
    num_primes, last_prime = count_primes(n)  # Calling the function to determine prime numbers
    print(f"There are {num_primes} prime numbers less than or equal to {n}.")  # Output of the number of prime numbers
    print(f"The last prime number is {last_prime}.")  # Output of the largest prime factor of n
    end = time.perf_counter()  # Ending time measurement
    duration = (end - begin) * 1000  # Calculating the duration of the time measurement in milliseconds
    print(f"Duration: {duration} ms")  # Output of the duration of the time measurement
