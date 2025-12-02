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
__global__ void dropout_kernel_optimized(
    T* output,
    const T* input,
    const uint8_t* mask,
    const int64_t numel,
    const float scale
) {
    const int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < numel) {
        uint8_t m0 = mask[idx];
        uint8_t m1 = mask[idx + 1];
        uint8_t m2 = mask[idx + 2];
        uint8_t m3 = mask[idx + 3];
        
        T i0 = input[idx];
        T i1 = input[idx + 1];
        T i2 = input[idx + 2];
        T i3 = input[idx + 3];
        
        output[idx] = m0 ? static_cast<T>(i0 * scale) : static_cast<T>(0);
        output[idx + 1] = m1 ? static_cast<T>(i1 * scale) : static_cast<T>(0);
        output[idx + 2] = m2 ? static_cast<T>(i2 * scale) : static_cast<T>(0);
        output[idx + 3] = m3 ? static_cast<T>(i3 * scale) : static_cast<T>(0);
    } else {
        for (int i = 0; i < 4 && idx + i < numel; i++) {
            output[idx + i] = mask[idx + i] ? static_cast<T>(input[idx + i] * scale) : static_cast<T>(0);
        }
    }
}

__global__ void generate_dropout_mask_kernel_optimized(
    uint8_t* mask,
    const int64_t numel,
    const float keep_prob,
    const uint64_t seed,
    const uint64_t offset
) {
    const int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < numel) {
        uint64_t hash0 = seed ^ (offset + idx);
        uint64_t hash1 = seed ^ (offset + idx + 1);
        uint64_t hash2 = seed ^ (offset + idx + 2);
        uint64_t hash3 = seed ^ (offset + idx + 3);
        
        float r0 = hash_to_float(hash0);
        float r1 = hash_to_float(hash1);
        float r2 = hash_to_float(hash2);
        float r3 = hash_to_float(hash3);
        
        mask[idx] = (r0 < keep_prob) ? 1 : 0;
        mask[idx + 1] = (r1 < keep_prob) ? 1 : 0;
        mask[idx + 2] = (r2 < keep_prob) ? 1 : 0;
        mask[idx + 3] = (r3 < keep_prob) ? 1 : 0;
    } else {
        for (int i = 0; i < 4 && idx + i < numel; i++) {
            uint64_t hash_input = seed ^ (offset + idx + i);
            float rand_val = hash_to_float(hash_input);
            mask[idx + i] = (rand_val < keep_prob) ? 1 : 0;
        }
    }
}

template <typename T>
__global__ void dropout_inplace_kernel_optimized(
    T* data,
    const int64_t numel,
    const float keep_prob,
    const float scale,
    const uint64_t seed,
    const uint64_t offset
) {
    const int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < numel) {
        uint64_t hash0 = seed ^ (offset + idx);
        uint64_t hash1 = seed ^ (offset + idx + 1);
        uint64_t hash2 = seed ^ (offset + idx + 2);
        uint64_t hash3 = seed ^ (offset + idx + 3);
        
        float r0 = hash_to_float(hash0);
        float r1 = hash_to_float(hash1);
        float r2 = hash_to_float(hash2);
        float r3 = hash_to_float(hash3);
        
        T d0 = data[idx];
        T d1 = data[idx + 1];
        T d2 = data[idx + 2];
        T d3 = data[idx + 3];
        
        data[idx] = (r0 < keep_prob) ? static_cast<T>(d0 * scale) : static_cast<T>(0);
        data[idx + 1] = (r1 < keep_prob) ? static_cast<T>(d1 * scale) : static_cast<T>(0);
        data[idx + 2] = (r2 < keep_prob) ? static_cast<T>(d2 * scale) : static_cast<T>(0);
        data[idx + 3] = (r3 < keep_prob) ? static_cast<T>(d3 * scale) : static_cast<T>(0);
    } else {
        for (int i = 0; i < 4 && idx + i < numel; i++) {
            uint64_t hash_input = seed ^ (offset + idx + i);
            float rand_val = hash_to_float(hash_input);
            data[idx + i] = (rand_val < keep_prob) ? static_cast<T>(data[idx + i] * scale) : static_cast<T>(0);
        }
    }
}

// ============ HOST CODE ============

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
    const int blocks = (numel + threads * 4 - 1) / (threads * 4);
    
    if (inplace) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dropout_inplace", ([&] {
            dropout_inplace_kernel_optimized<scalar_t><<<blocks, threads>>>(
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
        
        generate_dropout_mask_kernel_optimized<<<blocks, threads>>>(
            mask.data_ptr<uint8_t>(),
            numel,
            keep_prob,
            seed,
            offset
        );
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dropout", ([&] {
            dropout_kernel_optimized<scalar_t><<<blocks, threads>>>(
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