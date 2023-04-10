'''
Program Name: sieve_eratosthenes_multiprocessing.py
Author: Thomas Knoefel
Short Description: This program calculates the number of prime numbers and the largest prime factor of n.
Algorithm: Sieve of Eratosthenes
Optimization Method: Multiprocessing
Created on: February 20, 2023
Long Description: This program uses the Sieve of Eratosthenes to find all prime numbers up to a given value n.
'''

import time
import numpy as np
import multiprocessing

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

# Function for parallel execution of count_primes on a subinterval
def count_primes_parallel(start, end, result_queue):
    # Calculate the prime numbers in the given subinterval
    num_primes, last_prime = count_primes(end)
    if start > 2:
        start_num_primes, _ = count_primes(start - 1)
        num_primes -= start_num_primes
    # Adding the result to the result list
    result_queue.put((num_primes, last_prime))

if __name__ == "__main__":
    begin = time.perf_counter()  # Starting time measurement
    n = 100000000  # Determining the upper limit n for the search for prime numbers
    num_processes = multiprocessing.cpu_count()  # Determining the number of processes (usually the number of CPU cores)
    chunk_size = n // num_processes  # Calculating the size of each subinterval
    result_queue = multiprocessing.Queue()  # Creating a queue to store the results
    processes = []  # Creating a list of processes

    # Starting a process for each subinterval
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size - 1
        p = multiprocessing.Process(target=count_primes_parallel, args=(start, end, result_queue))
        p.start()
        processes.append(p)

    # Collecting the results and aggregating the number of prime numbers and the largest prime factor
    num_primes = 0
    last_prime = 0
    for i in range(num_processes):
        p = processes[i]
        p.join()
        (p_num_primes, p_last_prime) = result_queue.get()
        num_primes += p_num_primes
        last_prime = max(last_prime, p_last_prime)

    # Output of the number of prime numbers, the largest prime factor, and the duration of the calculation
    print(f"There are {num_primes} prime numbers less than or equal to {n}.")
    print(f"The last prime number is {last_prime}.")
    end = time.perf_counter()  # End time of the time measurement
    duration = (end - begin) * 1000  # Calculating the duration of the time measurement in milliseconds
    print(f"Duration: {duration} ms")
