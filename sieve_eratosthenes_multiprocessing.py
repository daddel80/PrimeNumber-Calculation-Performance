'''
Program Name: sieve_eratosthenes_multiprocessing.py
Author: Thomas Knoefel
Short Description: This program calculates the number of prime numbers and the largest prime factor of n.
Algorithm: Sieve of Eratosthenes
Optimization Method: Multiprocessing
Created on: February 20, 2023
Long Description: This program uses the Sieve of Eratosthenes to find all prime numbers up to a given value n, optimized with multiprocessing.
'''

import time
import numpy as np
import multiprocessing

# Definition of the function to determine the number of prime numbers
def count_primes(n):
    is_prime = np.ones(n, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            is_prime[i * i:n:i] = False
    num_primes = np.sum(is_prime)
    last_prime = np.max(np.nonzero(is_prime)[0]) if np.any(is_prime) else 0
    return num_primes, last_prime

# Function for parallel execution of count_primes on a subinterval
def count_primes_parallel(start, end, result_queue):
    process_start_time = time.process_time()
    num_primes_end, last_prime_end = count_primes(end)
    num_primes_start, _ = count_primes(start - 1) if start > 2 else (0, 0)
    num_primes = num_primes_end - num_primes_start
    process_time = time.process_time() - process_start_time
    result_queue.put((num_primes, last_prime_end, process_time * 1000))  # Convert to ms

if __name__ == "__main__":
    n = 100000000
    num_processes = multiprocessing.cpu_count()
    chunk_size = n // num_processes
    result_queue = multiprocessing.Queue()
    processes = []

    print(f"Number of processors used: {num_processes}")

    # Start time measurement
    begin_real = time.perf_counter()

    max_process_time = 0
    num_primes = 0
    last_prime = 0

    # Parallel process creation
    for i in range(num_processes):
        start = i * chunk_size + (2 if i == 0 else 1)  # Avoid "1" in the first chunk
        end = start + chunk_size - 1
        p = multiprocessing.Process(target=count_primes_parallel, args=(start, end, result_queue))
        p.start()
        processes.append(p)

    # Collecting results
    for p in processes:
        p.join()
        p_num_primes, p_last_prime, p_time = result_queue.get()
        num_primes += p_num_primes
        last_prime = max(last_prime, p_last_prime)
        max_process_time = max(max_process_time, p_time)  # Track the maximum process time

    # End time measurement
    end_real = time.perf_counter()

    # Calculate durations
    duration_real = (end_real - begin_real) * 1000  # Real-time in milliseconds

    # Output results
    print(f"There are {num_primes} prime numbers less than or equal to {n}.")
    print(f"The last prime number is {last_prime}.")
    
    # Convert durations to hh:mm:ss.ms
    def format_duration(duration):
        hours = int(duration // (3600 * 1000))
        minutes = int((duration % (3600 * 1000)) // (60 * 1000))
        seconds = int((duration % (60 * 1000)) // 1000)
        milliseconds = int(duration % 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

    # Output the formatted durations
    print(f"Duration (Real Time): {duration_real:.2f} ms ({format_duration(duration_real)})")
    print(f"Maximum Process Time: {max_process_time:.2f} ms ({format_duration(max_process_time)})")
