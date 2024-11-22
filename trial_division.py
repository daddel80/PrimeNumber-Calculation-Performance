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

# Function to determine the number of primes and the largest prime
def count_primes(n):
    count = 1  # Counter for prime numbers
    largest_prime = 2  # Start with the smallest prime number
    
    # Iterate through odd numbers from 3 to n
    for i in range(3, n + 1, 2):
        prime_found = True  # Assume number is prime
        # Check divisors up to the square root of the current number
        for j in range(3, isqrt(i) + 1, 2):
            if i % j == 0:  # If divisible, it's not prime
                prime_found = False
                break
        if prime_found:
            largest_prime = i  # Update largest prime
            count += 1  # Increment count
    
    return count, largest_prime  # Return number of primes and largest prime

if __name__ == "__main__":
    # Start time measurement
    begin_real = time.perf_counter()  # Wall clock time
    begin_cpu = time.process_time()  # CPU time
    
    varMAX = 100000000  # Define the upper limit for the prime search
    
    # Call the function to count primes
    num_primes, last_prime = count_primes(varMAX)
    
    # End time measurement
    end_real = time.perf_counter()  # Wall clock time
    end_cpu = time.process_time()  # CPU time
    
    # Calculate durations
    duration_real = (end_real - begin_real) * 1000  # Wall clock time in milliseconds
    duration_cpu = (end_cpu - begin_cpu) * 1000  # CPU time in milliseconds
    
    # Output results
    print(f"There are {num_primes} prime numbers less than or equal to {varMAX}.")  # Output number of primes
    print(f"The last prime number is {last_prime}.")  # Output the largest prime number
    
    # Convert durations to hh:mm:ss.ms
    def format_duration(duration):
        hours = int(duration // (3600 * 1000))  # Calculate hours
        minutes = int((duration % (3600 * 1000)) // (60 * 1000))  # Calculate minutes
        seconds = int((duration % (60 * 1000)) // 1000)  # Calculate seconds
        milliseconds = int(duration % 1000)  # Calculate milliseconds
        return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"  # Format as hh:mm:ss.ms

    # Output durations
    print(f"Duration (Real Time): {duration_real:.2f} ms ({format_duration(duration_real)})")  # Real time
    print(f"Duration (System Time): {duration_cpu:.2f} ms ({format_duration(duration_cpu)})")  # CPU time
