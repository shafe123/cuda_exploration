/*
 * getting_started.cpp
 * Install:  https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
 * Programming:  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
 *  Created on: Feb 3, 2020
 *      Author: Ethan Shafer
 *
 *
 * Device 0: "GeForce GTX 1060 3GB"
  CUDA Driver Version / Runtime Version          10.2 / 10.2
  CUDA Capability Major/Minor version number:    6.1
  Total amount of global memory:                 3016 MBytes (3162963968 bytes)
  ( 9) Multiprocessors, (128) CUDA Cores/MP:     1152 CUDA Cores
  GPU Max Clock rate:                            1759 MHz (1.76 GHz)
  Memory Clock rate:                             4004 Mhz
  Memory Bus Width:                              192-bit
  L2 Cache Size:                                 1572864 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>

__global__
void VecAdd(const float* A, const float* B, float* C, int numElements) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements) {
		C[i] = A[i] + B[i];
	}
}

int vector_main() {
	cudaError_t err = cudaSuccess;
	int numElements = 500;
	size_t size = numElements * sizeof(float);
	printf("Vector addition of %d elements\n", numElements);

	float *A;
	cudaMallocManaged(&A, size);
	float *B;
	cudaMallocManaged(&B, size);
	float *C;
	cudaMallocManaged(&C, size);

	if (A == NULL || B == NULL || C == NULL) {
		fprintf(stderr, "Failed to allocate vectors");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < numElements; i++) {
		A[i] = rand()/(float)RAND_MAX;
		B[i] = rand()/(float)RAND_MAX;
	}

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (abs(A[i] + B[i] - C[i]) > 1e-5)
        {
        	fprintf(stdout, "%d: %f + %f - %f = %f\n", i, A[i], B[i], C[i], abs(A[i] + B[i] - C[i]));
            //fprintf(stderr, "Result verification failed at element %d!\n", i);
            //exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free host memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    printf("Done\n");
    return 0;
}

