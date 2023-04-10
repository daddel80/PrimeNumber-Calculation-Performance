# Performance Test Programs

This repository contains a collection of performance test programs designed to measure the efficiency of prime number algorithms on different hardware and software platforms. The tests cover two different algorithms: the Sieve of Eratosthenes and the Trial Division method. Additionally, there are variants that use multiprocessing or GPU computing to further improve performance.

Please refer to the individual program descriptions for a list of required libraries and dependencies.

## Programs

The following table provides a summary of the different programs available in this repository:

| Program Name                              | Algorithm              | Graphics Card  | Multiprocessing |
| ----------------------------------------- | ---------------------- | -------------- | --------------- |
| `numpy_prime_number_finder.py`            | Sieve of Eratosthenes   | -              | -               |
| `cuda_prime_number_calculator.py`         | Sieve of Eratosthenes   | PyCUDA         | -               |
| `multiprocessing_prime_number_finder.py`  | Sieve of Eratosthenes   | -              | Yes             |
| `trial_division_prime_number_finder.py`   | Trial Division         | -              | -               |
| `cuda_trial_division_prime_number_finder.py` | Trial Division | -              | Yes             |
| `multiprocessing_trial_division_prime_number_finder.py` | Trial Division | -              | Yes             |

