# Performance Test Programs

This repository contains a collection of performance test programs designed to measure the efficiency of prime number algorithms on different hardware and software platforms. The tests cover two different algorithms: the Sieve of Eratosthenes and the Trial Division method. Additionally, there are variants that use multiprocessing or GPU computing to further improve performance.

Please refer to the individual program descriptions for a list of required libraries and dependencies.

## Programs

The following table provides a summary of the different programs available in this repository:

| Program Name                              | Algorithm              | Graphics Card  | Multiprocessing |
| ----------------------------------------- | ---------------------- | -------------- | --------------- |
| `sieve_of_eratosthenes.py`                | Sieve of Eratosthenes  | -              | -               |
| `cuda_sieve_of_eratosthenes.py`           | Sieve of Eratosthenes  | PyCUDA         | -               |
| `multiprocessing_sieve_of_eratosthenes.py`| Sieve of Eratosthenes  | -              | Yes             |
| `trial_division.py`                       | Trial Division         | -              | -               |
| `cuda_trial_division.py`                  | Trial Division         | PyCUDA         | -               |
| `multiprocessing_trial_division.py`       | Trial Division         | -              | Yes             |

## Algorithms

### Sieve of Eratosthenes

The Sieve of Eratosthenes is an ancient algorithm for finding all prime numbers up to a given limit. It works by iteratively marking as composite (i.e., not prime) the multiples of each prime, starting from the multiples of 2. The algorithm can be easily implemented using an array or list.

**Example:**
To find all prime numbers up to 10:

1. Create a list of integers from 2 to 10: `[2, 3, 4, 5, 6, 7, 8, 9, 10]`
2. Start with the first number (2), mark it as prime, and remove all its multiples from the list: `[2, 3, 5, 7, 9]`
3. Move to the next unmarked number (3), mark it as prime, and remove all its multiples from the list: `[2, 3, 5, 7]`
4. Continue this process until all numbers in the list have been processed.

### Trial Division

Trial Division is a simple algorithm to test the primality of a number. It works by dividing the given number by all odd integers up to its square root. If any divisor is found, the number is not prime. If no divisor is found, the number is prime. In the provided implementation, the program iterates over all odd numbers between 3 and a given maximum value with a step of 2. Then, it checks whether each odd number is divisible by another odd number up to its square root.

**Example:**
To check if 11 is a prime number:

1. Iterate over all odd numbers between 3 and 11 with a step of 2: `[3, 5, 7, 9, 11]`
2. For each number, check if it's divisible by another odd number up to its square root:
   - For 11, find the square root of 11, which is approximately 3.32.
   - Divide 11 by odd numbers up to 3, e.g., 3.
   - No divisor is found, so 11 is a prime number.

