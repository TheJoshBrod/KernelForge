```cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

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
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    
    for (; idx < numel; idx += stride) {
        uint8_t m = mask[idx];
        T val = input[idx];
        output[idx] = m ? static_cast<T>(val * scale) : static_cast<T>(0);
    }
}

__global__ void generate_dropout_mask_kernel(
    uint8_t* mask,
    const int64_t numel,
    const float keep_prob,
    const uint64_t seed,
    const uint64_t offset
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    
    for (; idx < numel; idx += stride) {
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
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    
    for (; idx < numel; idx += stride) {
        uint64_t hash_input = seed ^ (offset + idx);
        float rand_val = hash_to_float(hash_input);
        T val = data[idx];
        data[idx] = (rand_val < keep_prob) ? static_cast<T>(val * scale) : static_cast<T>(0);
    }
}

template <typename T, int VEC_SIZE>
__global__ void dropout_inplace_vectorized_kernel(
    T* data,
    const int64_t numel,
    const float keep_prob,
    const float scale,
    const uint64_t seed,
    const uint64_t offset
) {
    int64_t base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;
    const int64_t stride = blockDim.x * gridDim.x * VEC_SIZE;
    
    for (; base_idx < numel; base_idx += stride) {
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            int64_t idx = base_idx + i;
            if (idx < numel) {
                uint64_t hash_input = seed ^ (offset + idx);
                float rand_val = hash_to_float(hash_input);
                T val = data[idx];
                data[idx] = (rand_val < keep_prob) ? static_cast<T>(val * scale) : static_cast<T>(0);
            }
        }
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
    const int max_blocks = 82 * 4;
    
    if (inplace) {
        const int vec_size = 4;
        const int blocks = min(max_blocks, (int)((numel + vec_size * threads - 1) / (vec_size * threads)));
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dropout_inplace", ([&] {
            if (numel >= vec_size * threads) {
                dropout_inplace_vectorized_kernel<scalar_t, 4><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(),
                    numel,
                    keep_prob,
                    scale,
                    seed,
                    offset
                );
            } else {
                const int simple_blocks = min(max_blocks, (int)((numel + threads - 1) / threads));
                dropout_inplace_kernel<scalar_t><<<simple_blocks, threads>>>(
                    input.data_ptr<scalar_t>(),
                    numel,
                    keep_prob,
                    scale,
                    seed,
                    offset
                );
            }
        }));
        
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
        
        return input;
    } else {
        auto output = torch::empty_like(input);
        auto mask = torch::empty({numel}, torch::dtype(torch::kUInt8).device(input.device()));
        
        const int blocks = min(max_blocks, (int)((numel + threads - 1) / threads));
        
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
```