// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ============ DEVICE CODE (CUDA kernels only) ============

__device__ __forceinline__ float hash_to_float(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (x & 0xFFFFFFU) / 16777216.0f;
}

template <typename T>
__global__ void dropout_kernel(
    T* output,
    const T* input,
    const uint8_t* mask,
    const int64_t numel,
    const float scale
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = mask[idx] ? static_cast<T>(input[idx] * scale) : static_cast<T>(0);
    }
}

__global__ void generate_dropout_mask_kernel(
    uint8_t* mask,
    const int64_t numel,
    const float keep_prob,
    const uint64_t seed,
    const uint64_t offset
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        uint64_t hash_input = seed ^ (offset + idx);
        float rand_val = hash_to_float(hash_input);
        mask[idx] = (rand_val < keep_prob) ? 1 : 0;
    }
}

template <typename T>
__global__ void dropout_inplace_kernel(
    T* data,
    const int64_t numel,
    const float keep_prob,
    const float scale,
    const uint64_t seed,
    const uint64_t offset
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        uint64_t hash_input = seed ^ (offset + idx);
        float rand_val = hash_to_float(hash_input);
        data[idx] = (rand_val < keep_prob) ? static_cast<T>(data[idx] * scale) : static_cast<T>(0);
    }
}

// ============ HOST CODE ============

torch::Tensor launch(
    torch::Tensor input,
    double p,
    bool training,
    bool inplace
) {
    // 1. Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(p >= 0.0 && p <= 1.0, "dropout probability must be in [0, 1]");
    
    // 2. If not training, return input as-is (no dropout)
    if (!training || p == 0.0) {
        if (inplace) {
            return input;
        } else {
            return input.clone();
        }
    }
    
    // 3. If p == 1.0, return zeros
    if (p == 1.0) {
        if (inplace) {
            input.zero_();
            return input;
        } else {
            return torch::zeros_like(input);
        }
    }
    
    // 4. Calculate parameters
    const float keep_prob = 1.0f - static_cast<float>(p);
    const float scale = 1.0f / keep_prob;
    const int64_t numel = input.numel();
    
    // 5. Generate random seed
    auto gen = at::cuda::detail::getDefaultCUDAGenerator();
    auto philox_args = at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_engine_inputs(numel);
    const uint64_t seed = philox_args.first;
    const uint64_t offset = philox_args.second;
    
    // 6. Kernel launch parameters
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    
    // 7. Perform dropout
    if (inplace) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dropout_inplace", ([&] {
            dropout_inplace_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                numel,
                keep_prob,
                scale,
                seed,
                offset
            );
        }));
        
        // Check for errors
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
        
        return input;
    } else {
        auto output = torch::empty_like(input);
        
        // Generate mask
        auto mask = torch::empty({numel}, torch::dtype(torch::kUInt8).device(input.device()));
        
        generate_dropout_mask_kernel<<<blocks, threads>>>(
            mask.data_ptr<uint8_t>(),
            numel,
            keep_prob,
            seed,
            offset
        );
        
        // Apply dropout with mask
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dropout", ([&] {
            dropout_kernel<scalar_t><<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                mask.data_ptr<uint8_t>(),
                numel,
                scale
            );
        }));
        
        // Check for errors
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
        
        return output;
    }
}

// [END kernel.cu]