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
    begin_real = time.perf_counter()
    begin_cpu = time.process_time()
    
    n = 100000000
    num_primes, last_prime = count_primes(n)
    
    end_real = time.perf_counter()
    end_cpu = time.process_time()

    duration_real = (end_real - begin_real) * 1000
    duration_cpu = (end_cpu - begin_cpu) * 1000

    print(f"There are {num_primes} prime numbers less than or equal to {n}.")
    print(f"The last prime number is {last_prime}.")

    def format_duration(duration):
        hours = int(duration // (3600 * 1000))
        minutes = int((duration % (3600 * 1000)) // (60 * 1000))
        seconds = int((duration % (60 * 1000)) // 1000)
        milliseconds = int(duration % 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

    print(f"Duration (Real Time): {duration_real:.2f} ms ({format_duration(duration_real)})")
    print(f"Duration (System Time): {duration_cpu:.2f} ms ({format_duration(duration_cpu)})")