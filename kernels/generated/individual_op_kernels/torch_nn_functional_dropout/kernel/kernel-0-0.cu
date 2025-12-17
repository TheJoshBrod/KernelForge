// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <torch/extension.h>

// ============ DEVICE CODE (CUDA kernels only) ============

template <typename T>
__global__ void dropout_kernel(
    T* output,
    const T* input,
    unsigned char* mask,
    const int64_t numel,
    const float p,
    const float scale,
    const unsigned long long seed,
    const unsigned long long offset
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numel) {
        // Initialize random state
        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, offset, &state);
        
        // Generate random number
        float rand = curand_uniform(&state);
        
        // Apply dropout
        bool keep = rand >= p;
        mask[idx] = keep ? 1 : 0;
        output[idx] = keep ? static_cast<T>(static_cast<float>(input[idx]) * scale) : static_cast<T>(0);
    }
}

template <typename T>
__global__ void dropout_inference_kernel(
    T* output,
    const T* input,
    const int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numel) {
        output[idx] = input[idx];
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
    
    // Create mask tensor for gradient computation
    auto mask = torch::empty({numel}, torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
    
    // Scale factor to maintain expected value
    float scale = 1.0f / (1.0f - static_cast<float>(p));
    
    // Generate random seed
    auto gen = at::cuda::detail::getDefaultCUDAGenerator();
    unsigned long long seed = gen.random64();
    unsigned long long offset = 0;
    
    // 6. Launch kernel with dtype dispatch
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_contig.scalar_type(), "dropout_kernel", ([&] {
        dropout_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input_contig.data_ptr<scalar_t>(),
            mask.data_ptr<unsigned char>(),
            numel,
            static_cast<float>(p),
            scale,
            seed,
            offset
        );
    }));
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "dropout_kernel failed: ", cudaGetErrorString(err));
    
    // 7. Return tensor
    return output;
}

// [END kernel.cu]