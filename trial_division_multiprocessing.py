'''
Program Name: trial_division_multiprocessing.py
Author: Thomas Knoefel
Short Description: This program finds the number of prime numbers and the largest prime number within a specified range using the Trial Division method and multiprocessing.
Algorithm: Trial Division
Optimization Method: Multiprocessing
Created on: February 20, 2023
Long Description: This program uses the trial division algorithm to find all prime numbers within a range, optimized with multiprocessing.
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

# Function to count primes in a given range
def count_primes(start, end):
    count = 0
    largest_prime = 0
    for i in range(start, end):
        if is_prime(i):
            count += 1
            largest_prime = i
    return count, largest_prime

# Function for parallel execution of count_primes on a subinterval
def count_primes_parallel(start, end, result_queue):
    process_start_time = time.process_time()
    num_primes, last_prime = count_primes(start, end)
    process_time = time.process_time() - process_start_time
    result_queue.put((num_primes, last_prime, process_time * 1000))  # Convert to milliseconds

if __name__ == "__main__":
    varMAX = 100000000  # Define the upper limit for the prime search
    num_processes = multiprocessing.cpu_count()  # Get the number of processors
    chunk_size = varMAX // num_processes
    result_queue = multiprocessing.Queue()
    processes = []

    print(f"Number of processors used: {num_processes}")

    # Start real-time measurement
    begin_real = time.perf_counter()

    total_count = 0
    largest_prime = 0
    max_process_time = 0

    # Create and start processes
    for i in range(num_processes):
        start = 3 + i * chunk_size
        end = start + chunk_size
        p = multiprocessing.Process(target=count_primes_parallel, args=(start, end, result_queue))
        p.start()
        processes.append(p)

    # Collect results
    for i in range(num_processes):
        p = processes[i]
        p.join()
        p_count, p_largest_prime, p_time = result_queue.get()
        total_count += p_count
        largest_prime = max(largest_prime, p_largest_prime)
        max_process_time = max(max_process_time, p_time)  # Keep track of the longest process time

    # End real-time measurement
    end_real = time.perf_counter()

    # Calculate durations
    duration_real = (end_real - begin_real) * 1000  # Real-time in milliseconds

    # Output results
    print(f"There are {total_count} prime numbers less than or equal to {varMAX}.")
    print(f"The last prime number is {largest_prime}.")
    
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
