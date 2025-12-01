#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <chrono>

// ============ DEVICE CODE (CUDA kernels only) ============

// Optimized dropout kernel using vectorized loads for GTX 1660 Ti
template <typename T>
__global__ void dropout_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    unsigned char* __restrict__ mask,
    const int64_t numel,
    const float p,
    const float scale,
    const unsigned long long seed,
    const unsigned long long offset) {
    
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    
    // Initialize random state per thread
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, offset, &state);
    
    // Process multiple elements per thread for better memory bandwidth utilization
    for (int64_t i = idx; i < numel; i += stride) {
        float rand_val = curand_uniform(&state);
        bool keep = rand_val >= p;
        
        if (mask != nullptr) {
            mask[i] = keep ? 1 : 0;
        }
        
        output[i] = keep ? static_cast<T>(static_cast<float>(input[i]) * scale) : static_cast<T>(0);
    }
}

// Specialized kernel for training mode (with mask)
template <typename T>
__global__ void dropout_train_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    unsigned char* __restrict__ mask,
    const int64_t numel,
    const float p,
    const float scale,
    const unsigned long long seed) {
    
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);
    
    for (int64_t i = idx; i < numel; i += stride) {
        float rand_val = curand_uniform(&state);
        bool keep = rand_val >= p;
        
        mask[i] = keep ? 1 : 0;
        output[i] = keep ? static_cast<T>(static_cast<float>(input[i]) * scale) : static_cast<T>(0);
    }
}

// Specialized kernel for inference mode (no dropout, just copy)
template <typename T>
__global__ void dropout_inference_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    const int64_t numel) {
    
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = input[i];
    }
}

// ============ HOST CODE ============

torch::Tensor launch(torch::Tensor input, double p, bool training, bool inplace) {
    // 1. Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(p >= 0.0 && p <= 1.0, "dropout probability must be in [0, 1]");
    
    // 2. Handle edge cases
    if (p == 0.0 || !training) {
        // No dropout: return input as-is or copy
        if (inplace) {
            return input;
        } else {
            return input.clone();
        }
    }
    
    if (p == 1.0) {
        // Drop everything
        if (inplace) {
            input.zero_();
            return input;
        } else {
            return torch::zeros_like(input);
        }
    }
    
    // 3. Output tensor creation
    torch::Tensor output = inplace ? input : torch::empty_like(input);
    
    // 4. Calculate scale factor (1 / (1 - p)) for inverted dropout
    float scale = 1.0f / (1.0f - static_cast<float>(p));
    
    // 5. Get total number of elements
    int64_t numel = input.numel();
    
    // 6. Kernel launch parameters optimized for GTX 1660 Ti
    // GTX 1660 Ti has 24 SMs, 1536 cores, so use 256 threads/block
    const int threads = 256;
    const int blocks = std::min((numel + threads - 1) / threads, (int64_t)1024);
    
    // 7. Generate random seed using high-resolution clock
    unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    // 8. Launch kernel with dtype dispatch
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dropout_cuda", ([&] {
        if (training) {
            dropout_train_kernel<scalar_t><<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                nullptr,  // mask not needed for this simple implementation
                numel,
                static_cast<float>(p),
                scale,
                seed
            );
        } else {
            dropout_inference_kernel<scalar_t><<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                numel
            );
        }
    }));
    
    // 9. Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    // 10. Return tensor
    return output;
}

// [END kernel.cu]