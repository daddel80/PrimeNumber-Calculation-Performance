'''
Program Name: trial_division_cuda.py
Author: Thomas Knoefel
Short Description: This program calculates the count of prime numbers and the largest prime factor of n.
Algorithm: Trial Division
Optimization Method: CUDA
Created on: 10 April 2023
Long Description: This program uses a custom algorithm and CUDA to find all prime numbers up to a given value n.
'''

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time

# CUDA kernel function to count prime numbers
mod = SourceModule("""
    __device__ float my_sqrt(float x) {
        return sqrtf(x);
    }
    
    __global__ void count_primes(int *output, int *last_prime, int n) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        int num_primes = 0;
        int my_last_prime = 0;
        for (int i = 3 + 2 * tid; i <= n; i += 2 * stride) {
            int sqrt_i = my_sqrt(i) + 1;
            bool is_prime = true;
            for (int j = 3; j <= sqrt_i; j += 2) {
                if (i % j == 0) {
                    is_prime = false;
                    break;
                }
            }
            if (is_prime) {
                num_primes++;
                my_last_prime = i;
            }
        }

        // Reduce num_primes and my_last_prime across threads in block
        __shared__ int shared_num_primes[256];
        __shared__ int shared_last_prime[256];
        shared_num_primes[threadIdx.x] = num_primes;
        shared_last_prime[threadIdx.x] = my_last_prime;
        for (int i = blockDim.x / 2; i > 0; i >>= 1) {
            __syncthreads();
            if (threadIdx.x < i) {
                shared_num_primes[threadIdx.x] += shared_num_primes[threadIdx.x + i];
                if (shared_last_prime[threadIdx.x + i] > shared_last_prime[threadIdx.x]) {
                    shared_last_prime[threadIdx.x] = shared_last_prime[threadIdx.x + i];
                }
            }
        }

        // Perform a reduction across blocks
        if (threadIdx.x == 0) {
            for (int i = 1; i < blockDim.x; i++) {
                shared_num_primes[0] += shared_num_primes[i];
                if (shared_last_prime[i] > shared_last_prime[0]) {
                    shared_last_prime[0] = shared_last_prime[i];
                }
            }
            atomicAdd(output, shared_num_primes[0]);
            atomicMax(last_prime, shared_last_prime[0]);
        }
    }
""")

def count_primes(n):
    # Allocate output arrays on host and device
    output_host = np.array([1], dtype=np.int32)
    last_prime_host = np.array([2], dtype=np.int32)
    output_device = cuda.mem_alloc(output_host.nbytes)
    last_prime_device = cuda.mem_alloc(last_prime_host.nbytes)
    cuda.memcpy_htod(output_device, output_host)
    cuda.memcpy_htod(last_prime_device, last_prime_host)

    # Calculate kernel parameters
    block_size = 256
    grid_size = (n - 3) // (2 * block_size) + 1
    print(f"block_size: {block_size}, grid_size: {grid_size}")

    # Run kernel function
    count_primes_kernel = mod.get_function("count_primes")
    count_primes_kernel(output_device, last_prime_device, np.int32(n), block=(block_size, 1, 1), grid=(grid_size, 1))
    cuda.memcpy_dtoh(output_host, output_device)
    cuda.memcpy_dtoh(last_prime_host, last_prime_device)

    return output_host[0], last_prime_host[0]

if __name__ == "__main__":
    begin = time.perf_counter()
    n = 100000000
    num_primes, last_prime = count_primes(n)
    print(f"There are {num_primes} prime numbers less than or equal to {n}.")
    print(f"The last prime number is {last_prime}.")
    end = time.perf_counter()
    duration = (end - begin) * 1000
    print(f"Duration: {duration} ms")