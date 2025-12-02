#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__device__ __forceinline__ float hash_to_float(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (x & 0xFFFFFFU) * 5.960464477539063e-8f;
}

template <typename T>
__global__ void dropout_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    const uint8_t* __restrict__ mask,
    const int64_t numel,
    const float scale
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    
    for (; idx < numel; idx += stride) {
        output[idx] = mask[idx] ? static_cast<T>(input[idx] * scale) : static_cast<T>(0);
    }
}

__global__ void generate_dropout_mask_kernel(
    uint8_t* __restrict__ mask,
    const int64_t numel,
    const float keep_prob,
    const uint64_t seed,
    const uint64_t offset
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    
    for (; idx < numel; idx += stride) {
        uint64_t hash_input = seed ^ (offset + idx);
        float rand_val = hash_to_float(hash_input);
        mask[idx] = (rand_val < keep_prob) ? 1 : 0;
    }
}

template <typename T>
__global__ void dropout_inplace_kernel(
    T* __restrict__ data,
    const int64_t numel,
    const float keep_prob,
    const float scale,
    const uint64_t seed,
    const uint64_t offset
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    
    for (; idx < numel; idx += stride) {
        uint64_t hash_input = seed ^ (offset + idx);
        float rand_val = hash_to_float(hash_input);
        data[idx] = (rand_val < keep_prob) ? static_cast<T>(data[idx] * scale) : static_cast<T>(0);
    }
}

torch::Tensor launch(
    torch::Tensor input,
    double p,
    bool training,
    bool inplace
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(p >= 0.0 && p <= 1.0, "dropout probability must be in [0, 1]");
    
    if (!training || p == 0.0) {
        if (inplace) {
            return input;
        } else {
            return input.clone();
        }
    }
    
    if (p == 1.0) {
        if (inplace) {
            input.zero_();
            return input;
        } else {
            return torch::zeros_like(input);
        }
    }
    
    const float keep_prob = 1.0f - static_cast<float>(p);
    const float scale = 1.0f / keep_prob;
    const int64_t numel = input.numel();
    
    const uint64_t seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const uint64_t offset = 0;
    
    const int threads = 256;
    const int blocks = min((numel + threads - 1) / threads, 2048L);
    
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
        
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
        
        return input;
    } else {
        auto output = torch::empty_like(input);
        auto mask = torch::empty({numel}, torch::dtype(torch::kUInt8).device(input.device()));
        
        generate_dropout_mask_kernel<<<blocks, threads>>>(
            mask.data_ptr<uint8_t>(),
            numel,
            keep_prob,
            seed,
            offset
        );
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dropout", ([&] {
            dropout_kernel<scalar_t><<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                mask.data_ptr<uint8_t>(),
                numel,
                scale
            );
        }));
        
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
        
        return output;
    }
}