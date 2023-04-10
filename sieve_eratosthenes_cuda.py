'''
Program Name: sieve_eratosthenes_cuda.py
Author: Thomas Knoefel
Short Description: This program calculates the number of prime numbers and the largest prime factor of n.
Algorithm: Sieve of Eratosthenes
Optimization Method: CUDA
Created on: February 20, 2023
Long Description: This program uses the Sieve of Eratosthenes to find all prime numbers up to a given value n.
'''

import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule
import time

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

def count_primes(n):
    # Allocate output arrays on host and device
    is_prime_host = np.ones(n, dtype=bool)
    is_prime_host[0:2] = False
    is_prime_device = cuda.mem_alloc(is_prime_host.nbytes)
    cuda.memcpy_htod(is_prime_device, is_prime_host)

    # Calculate kernel parameters
    block_size = 256
    grid_size = (n - 1) // (2 * block_size) + 1

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
    begin = time.perf_counter()
    n = 100000000
    num_primes, last_prime = count_primes(n)
    print(f"There are {num_primes} prime numbers less than or equal to {n}.")
    print(f"The last prime number is {last_prime}.")
    end = time.perf_counter()
    duration = (end - begin) * 1000
    print(f"Duration: {duration} ms")
