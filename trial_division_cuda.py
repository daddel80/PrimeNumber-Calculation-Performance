"""
Program Name: trial_division_cuda.py
Author: Thomas Knoefel
Short Description: This program calculates the count of prime numbers and the largest prime factor of n.
Algorithm: Trial Division
Optimization Method: CUDA
Created on: April 10, 2023
"""

import warnings
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time

# Suppress the specific UserWarning from PyCUDA
warnings.filterwarnings("ignore", category=UserWarning)

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
            int sqrt_i = (int)my_sqrt((float)i) + 1;
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

        __shared__ int shared_num_primes[256];
        __shared__ int shared_last_prime[256];
        shared_num_primes[threadIdx.x] = num_primes;
        shared_last_prime[threadIdx.x] = my_last_prime;

        __syncthreads();

        for (int i = blockDim.x / 2; i > 0; i >>= 1) {
            if (threadIdx.x < i) {
                shared_num_primes[threadIdx.x] += shared_num_primes[threadIdx.x + i];
                shared_last_prime[threadIdx.x] = max(shared_last_prime[threadIdx.x], shared_last_prime[threadIdx.x + i]);
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            atomicAdd(output, shared_num_primes[0]);
            atomicMax(last_prime, shared_last_prime[0]);
        }
    }
""")

def count_primes(n):
    output_host = np.array([1], dtype=np.int32)  # Include 2 as the first prime
    last_prime_host = np.array([2], dtype=np.int32)
    output_device = cuda.mem_alloc(output_host.nbytes)
    last_prime_device = cuda.mem_alloc(last_prime_host.nbytes)
    cuda.memcpy_htod(output_device, output_host)
    cuda.memcpy_htod(last_prime_device, last_prime_host)

    block_size = 256
    grid_size = (n - 3) // (2 * block_size) + 1
    print(f"block_size: {block_size}, grid_size: {grid_size}")

    count_primes_kernel = mod.get_function("count_primes")
    count_primes_kernel(output_device, last_prime_device, np.int32(n), block=(block_size, 1, 1), grid=(grid_size, 1))
    cuda.memcpy_dtoh(output_host, output_device)
    cuda.memcpy_dtoh(last_prime_host, last_prime_device)

    return output_host[0], last_prime_host[0]

if __name__ == "__main__":
    begin_real = time.perf_counter()
    begin_cpu = time.process_time()

    varMAX = 100000000
    num_primes, last_prime = count_primes(varMAX)

    end_real = time.perf_counter()
    end_cpu = time.process_time()

    duration_real = (end_real - begin_real) * 1000
    duration_cpu = (end_cpu - begin_cpu) * 1000

    print(f"There are {num_primes} prime numbers less than or equal to {varMAX}.")
    print(f"The last prime number is {last_prime}.")

    def format_duration(duration):
        hours = int(duration // (3600 * 1000))
        minutes = int((duration % (3600 * 1000)) // (60 * 1000))
        seconds = int((duration % (60 * 1000)) // 1000)
        milliseconds = int(duration % 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

    print(f"Duration (Real Time): {duration_real:.2f} ms ({format_duration(duration_real)})")
    print(f"Duration (System Time): {duration_cpu:.2f} ms ({format_duration(duration_cpu)})")
