// compile: -std=c++17
// baseline_min_ms: 2000
// baseline_max_ms: 8000
// weight: 0.2
// description: Simple CUDA C++ compilation test with basic kernels

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Simple CUDA kernel for vector addition
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// CUDA kernel for matrix multiplication (simplified)
__global__ void matrixMul(const float* a, const float* b, float* c, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += a[row * width + k] * b[k * width + col];
        }
        c[row * width + col] = sum;
    }
}

// Device function for complex calculations
__device__ float deviceCalculation(float x, float y) {
    return sqrtf(x * x + y * y);
}

// Kernel using device function
__global__ void complexKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        float y = data[idx] * 0.5f;
        data[idx] = deviceCalculation(x, y);
    }
}

int main() {
    std::cout << "CUDA Simple Test Starting..." << std::endl;
    
    // Check CUDA device properties (compile-time only, won't execute)
    const int N = 1024;
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;
    
    std::cout << "Vector size: " << N << std::endl;
    std::cout << "Block size: " << blockSize << std::endl;
    std::cout << "Number of blocks: " << numBlocks << std::endl;
    
    // Matrix dimensions for matrix multiplication test
    const int matrixWidth = 32;
    const int matrixSize = matrixWidth * matrixWidth;
    
    std::cout << "Matrix width: " << matrixWidth << std::endl;
    std::cout << "Matrix size: " << matrixSize << std::endl;
    
    // Calculate grid dimensions for matrix multiplication
    dim3 blockDim(16, 16);
    dim3 gridDim((matrixWidth + blockDim.x - 1) / blockDim.x,
                 (matrixWidth + blockDim.y - 1) / blockDim.y);
    
    std::cout << "Matrix block dim: (" << blockDim.x << ", " << blockDim.y << ")" << std::endl;
    std::cout << "Matrix grid dim: (" << gridDim.x << ", " << gridDim.y << ")" << std::endl;
    
    // Test some CUDA-specific syntax and functions
    std::cout << "CUDA compilation test - demonstrating:" << std::endl;
    std::cout << "1. __global__ kernel functions" << std::endl;
    std::cout << "2. __device__ functions" << std::endl;
    std::cout << "3. CUDA built-in variables (blockIdx, threadIdx, etc.)" << std::endl;
    std::cout << "4. CUDA math functions (sqrtf)" << std::endl;
    std::cout << "5. dim3 data types" << std::endl;
    std::cout << "6. Memory access patterns" << std::endl;
    
    std::cout << "CUDA Simple Test Completed Successfully!" << std::endl;
    
    return 0;
}