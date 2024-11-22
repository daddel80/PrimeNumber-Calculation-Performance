# **Table of Contents**

- [Programs](#programs)
- [Algorithms](#algorithms)
   - [Sieve of Eratosthenes](#sieve-of-eratosthenes)
   - [Trial Division](#trial-division)
- [Performance Tests](#performance-tests)
   - [System Configuration](#system-configuration)
   - [Runtime Environment](#runtime-environment)
- [Performance Results](#performance-results)
- [Useful Links](#useful-links)

---

# **Prime Number Algorithm Performance**

This repository contains a collection of performance test programs designed to measure the efficiency of prime number algorithms on different hardware and software platforms. The tests cover two different algorithms: the Sieve of Eratosthenes and the Trial Division method. Additionally, there are variants that use multiprocessing or GPU computing to further improve performance.

Please refer to the individual program descriptions for a list of required libraries and dependencies.

---

## **Programs**

The following table provides a summary of the different programs available in this repository:

| Program Name                              | Algorithm              | Graphics Card  | Multiprocessing |
| ----------------------------------------- | ---------------------- | -------------- | --------------- |
| `sieve_of_eratosthenes.py`                | Sieve of Eratosthenes  | -              | -               |
| `multiprocessing_sieve_of_eratosthenes.py`| Sieve of Eratosthenes  | -              | Yes             |
| `cuda_sieve_of_eratosthenes.py`           | Sieve of Eratosthenes  | PyCUDA         | -               |
| `trial_division.py`                       | Trial Division         | -              | -               |
| `multiprocessing_trial_division.py`       | Trial Division         | -              | Yes             |
| `cuda_trial_division.py`                  | Trial Division         | PyCUDA         | -               |

---

## **Algorithms**

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

---

## **Performance Tests**

### System Configuration

The following performance tests were conducted with an upper limit of 100,000,000 for prime number computation. Tests were run on a system with the following configuration:

- **Operating System:** Windows 11  
- **Processor:** Intel i7-11800H @ 2.30GHz (8 Cores, 16 Threads)  
- **Memory:** 32 GB  
- **Graphics Card:** NVIDIA RTX A2000  

### **Runtime Environment**

To execute these programs, ensure the following components are installed:

1. **CUDA Development Kit**  
   - Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) matching your GPU and operating system.  
   - Add the `nvcc` compiler to your system's PATH.

2. **C Compiler (e.g., Visual Studio)**  
   - Install a C compiler compatible with CUDA, such as Visual Studio.  
   - During installation, enable the **Desktop Development with C++** workload.  
   - Ensure `cl.exe` is in the PATH. It is typically located in:  
     `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\<version>\bin\Hostx64\x64`.

3. **Python Dependencies**  
   - Install the required Python packages using `pip`:  
     ```bash
     pip install pycuda numpy
     ```

---

## **Performance Results**

| **Script**                                | **Real Time**  | **System Time**  | **Additional Info**                   |
|-------------------------------------------|----------------|------------------|----------------------------------------|
| `sieve_eratosthenes.py`                   | 00:07:45.452   | 00:07:34.343     | -                                      |
| `sieve_eratosthenes_multiprocessing.py`   | 00:08:51.862   | 00:07:41.687     | processors used: 16                   |
| `sieve_eratosthenes_cuda.py`              | 00:01:40.675   | 00:01:41.687     | block_size: 256, grid_size: 195313     |
| `trial_division.py`                       | 00:15:49.027   | 00:15:33.953     | -                                      |
| `trial_division_multiprocessing.py`       | 00:02:55.457   | 00:02:45.500     | processors used: 16                   |
| `trial_division_cuda.py`                  | 00:01:15.236   | 00:01:25.250     | block_size: 256, grid_size: 195313     |

> **Note:** In some cases, the **System Time** exceeds the **Real Time** due to asynchronous GPU computations and CPU overheads. The **System Time** includes time spent by the CPU managing GPU tasks and may not represent the exact duration of GPU computations.

---

## **Useful Links**

- [Official CUDA Toolkit (developer.nvidia.com)](https://developer.nvidia.com/cuda-toolkit)  
- [Python Package Installation (pypi.org)](https://pypi.org/project/pycuda/)  
- [Microsoft Visual Studio (visualstudio.microsoft.com)](https://visualstudio.microsoft.com/)  
