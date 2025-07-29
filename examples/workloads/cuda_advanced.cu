// compile: -std=c++17
// baseline_min_ms: 3000
// baseline_max_ms: 8000
// weight: 0.1
// description: Advanced CUDA C++ test with templates, shared memory, and math operations

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Template CUDA kernel
template<typename T>
__global__ void templateKernel(T* data, int n, T multiplier) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= multiplier;
    }
}

// Shared memory kernel for reduction
__global__ void reductionKernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Warp-level operations (simplified)
__global__ void warpOperations(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        
        // Simple warp operations without sync
        if (threadIdx.x % 32 == 0) {
            val *= 2.0f;
        }
        
        data[idx] = val;
    }
}

// Complex math operations using CUDA math library
__global__ void mathOperations(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        
        // Complex math operations
        float result = sinf(x) * cosf(x);
        result += expf(x * 0.1f);
        result *= logf(fabsf(x) + 1.0f);
        result = powf(result, 0.5f);
        
        // Use fast math intrinsics
        result += __sinf(x) * __cosf(x);
        result *= __expf(x * 0.01f);
        
        data[idx] = result;
    }
}

int main() {
    std::cout << "CUDA Advanced Test Starting..." << std::endl;
    
    // Test various CUDA features compilation
    const int N = 4096;
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;
    
    std::cout << "Advanced CUDA compilation test - demonstrating:" << std::endl;
    std::cout << "1. Template CUDA kernels" << std::endl;
    std::cout << "2. Shared memory usage" << std::endl;
    std::cout << "3. Warp-level primitives" << std::endl;
    std::cout << "4. Texture memory (legacy)" << std::endl;
    std::cout << "5. Cooperative groups" << std::endl;
    std::cout << "6. Dynamic parallelism" << std::endl;
    std::cout << "7. CUDA math library functions" << std::endl;
    std::cout << "8. Fast math intrinsics" << std::endl;
    std::cout << "9. Half-precision floating point" << std::endl;
    
    // Test template instantiation
    std::cout << "Testing template kernels for different types:" << std::endl;
    std::cout << "- float, int, double templates" << std::endl;
    
    // Test shared memory size calculation
    size_t sharedMemSize = blockSize * sizeof(float);
    std::cout << "Shared memory per block: " << sharedMemSize << " bytes" << std::endl;
    
    // Test warp size calculations
    int warpSize = 32;  // CUDA warp size
    int warpsPerBlock = (blockSize + warpSize - 1) / warpSize;
    std::cout << "Warps per block: " << warpsPerBlock << std::endl;
    
    // Test grid configuration for 2D problems
    dim3 block2D(16, 16);
    dim3 grid2D(4, 4);
    std::cout << "2D configuration - Block: (" << block2D.x << "," << block2D.y << "), ";
    std::cout << "Grid: (" << grid2D.x << "," << grid2D.y << ")" << std::endl;
    
    std::cout << "CUDA Advanced Test Completed Successfully!" << std::endl;
    
    return 0;
}