// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <chrono>

// ============ DEVICE CODE (CUDA kernels only) ============

// Simple inline random number generator (LCG-based hash)
__device__ inline float cuda_rand(unsigned long long seed, int64_t idx) {
    unsigned long long x = seed + idx;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return float(x & 0xFFFFFFFF) / float(0xFFFFFFFF);
}

template <typename T>
__global__ void dropout_kernel(
    T* output,
    const T* input,
    const int64_t numel,
    const float p,
    const float scale,
    const unsigned long long seed
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numel) {
        // Generate random number
        float rand = cuda_rand(seed, idx);
        
        // Apply dropout
        bool keep = rand >= p;
        output[idx] = keep ? static_cast<T>(static_cast<float>(input[idx]) * scale) : static_cast<T>(0);
    }
}

// ============ HOST CODE ============

torch::Tensor launch(torch::Tensor input, double p, bool training, bool inplace) {
    // 1. Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(p >= 0.0 && p <= 1.0, "dropout probability must be between 0 and 1");
    
    // Make input contiguous
    auto input_contig = input.contiguous();
    
    // 2. Output tensor creation
    torch::Tensor output;
    if (inplace) {
        output = input_contig;
    } else {
        output = torch::empty_like(input_contig);
    }
    
    // Get number of elements
    int64_t numel = input_contig.numel();
    
    // 3. If not training or p == 0, just copy input to output
    if (!training || p == 0.0) {
        if (!inplace) {
            output.copy_(input_contig);
        }
        return output;
    }
    
    // 4. If p == 1.0, zero out the output
    if (p == 1.0) {
        output.zero_();
        return output;
    }
    
    // 5. Kernel launch parameters
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    // Scale factor to maintain expected value
    float scale = 1.0f / (1.0f - static_cast<float>(p));
    
    // Generate random seed
    unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();
    
    // 6. Launch kernel with dtype dispatch
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_contig.scalar_type(), "dropout_kernel", ([&] {
        dropout_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input_contig.data_ptr<scalar_t>(),
            numel,
            static_cast<float>(p),
            scale,
            seed
        );
    }));
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "dropout_kernel failed: ", cudaGetErrorString(err));
    
    // 7. Return tensor
    return output;
}

// [END kernel.cu]