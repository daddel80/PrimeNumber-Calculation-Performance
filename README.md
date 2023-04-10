# Performance Test Programs

This repository contains a collection of performance test programs designed to measure the efficiency of prime number algorithms on different hardware and software platforms. The tests cover two different algorithms: the Sieve of Eratosthenes and the Trial Division method. Additionally, there are variants that use GPU computing and multiprocessing to further improve performance.

Please refer to the individual program descriptions for a list of required libraries and dependencies.

## Programs

The following table provides a summary of the different programs available in this repository:

| Program Name                                         | Algorithm             | Graphics Card  | Multiprocessing |
| ---------------------------------------------------- | --------------------- | -------------- | --------------- |
| `sieve_eratosthenes.py`                             | Sieve of Eratosthenes | -              | -               |
| `sieve_eratosthenes_cuda.py`                        | Sieve of Eratosthenes | CUDA           | -               |
| `sieve_eratosthenes_multiprocessing.py`             | Sieve of Eratosthenes | -              | Yes             |
| `trial_division.py`                                 | Trial Division        | -              | -               |
| `trial_division_cuda.py`                            | Trial Division        | CUDA           | -               |
| `trial_division_multiprocessing.py`                 | Trial Division        | -              | Yes             |


