'''
Program Name: sieve_eratosthenes_cuda.py
Author: Thomas Knoefel
Short Description: This program calculates the number of prime numbers and the largest prime factor of n.
Algorithm: Sieve of Eratosthenes
Optimization Method: CUDA
Created on: February 20, 2023
Long Description: This program uses the Sieve of Eratosthenes to find all prime numbers up to a given value n.
'''

import warnings
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule
import time

# Suppress the specific UserWarning from PyCUDA
warnings.filterwarnings("ignore", category=UserWarning)

mod = SourceModule("""
__global__ void count_primes_kernel(bool *is_prime, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Sieve of Eratosthenes
    int limit = sqrt((float)n);
    for (int i = 2 + tid; i <= limit; i += stride) {
        if (is_prime[i]) {
            for (int j = i * i; j <= n; j += i) {
                is_prime[j] = false;
            }
        }
    }
}
""")

# Grid size
block_size = 256

def count_primes(n):
    # Allocate output arrays on host and device
    is_prime_host = np.ones(n, dtype=bool)
    is_prime_host[0:2] = False
    is_prime_device = cuda.mem_alloc(is_prime_host.nbytes)
    cuda.memcpy_htod(is_prime_device, is_prime_host)

    # Calculate kernel parameters
    grid_size = (n - 3) // (2 * block_size) + 1
    print(f"block_size: {block_size}, grid_size: {grid_size}")

    # Run kernel function
    count_primes_kernel = mod.get_function("count_primes_kernel")
    count_primes_kernel(is_prime_device, np.int32(n), block=(block_size, 1, 1), grid=(grid_size, 1))

    # Copy results from device to host
    cuda.memcpy_dtoh(is_prime_host, is_prime_device)

    # Count primes
    num_primes = np.sum(is_prime_host)
    last_prime = np.max(np.where(is_prime_host))

    return num_primes, last_prime

if __name__ == "__main__":
    begin_real = time.perf_counter()
    begin_cpu = time.process_time()
    
    n = 100000000
    num_primes, last_prime = count_primes(n)
    
    end_real = time.perf_counter()
    end_cpu = time.process_time()

    duration_real = (end_real - begin_real) * 1000
    duration_cpu = (end_cpu - begin_cpu) * 1000
    
    # Calculate kernel parameters
    grid_size = (n - 1) // (2 * block_size) + 1

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
