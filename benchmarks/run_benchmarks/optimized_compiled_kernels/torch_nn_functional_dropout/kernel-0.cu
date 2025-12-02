#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <chrono>

__device__ __forceinline__ float hash_to_float(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (x & 0xFFFFFFU) * 5.960464477539063e-08f;
}

template <typename T, int VEC_SIZE>
__global__ void __launch_bounds__(256, 8) dropout_inplace_vectorized_kernel(
    T* data,
    const int64_t numel,
    const float keep_prob,
    const float scale,
    const uint64_t seed,
    const uint64_t offset
) {
    using VecT = typename std::conditional<VEC_SIZE == 4, float4,
                 typename std::conditional<VEC_SIZE == 2, float2, float>::type>::type;
    
    const int64_t vec_idx = (blockIdx.x * blockDim.x + threadIdx.x);
    const int64_t idx = vec_idx * VEC_SIZE;
    
    if (idx + VEC_SIZE <= numel) {
        VecT vec_data = *reinterpret_cast<VecT*>(&data[idx]);
        T* elem_ptr = reinterpret_cast<T*>(&vec_data);
        
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            uint64_t hash_input = seed ^ (offset + idx + i);
            float rand_val = hash_to_float(hash_input);
            elem_ptr[i] = (rand_val < keep_prob) ? static_cast<T>(elem_ptr[i] * scale) : static_cast<T>(0);
        }
        
        *reinterpret_cast<VecT*>(&data[idx]) = vec_data;
    } else if (idx < numel) {
        #pragma unroll
        for (int i = 0; i < VEC_SIZE && (idx + i) < numel; ++i) {
            uint64_t hash_input = seed ^ (offset + idx + i);
            float rand_val = hash_to_float(hash_input);
            data[idx + i] = (rand_val < keep_prob) ? static_cast<T>(data[idx + i] * scale) : static_cast<T>(0);
        }
    }
}

template <typename T>
__global__ void __launch_bounds__(256, 8) dropout_inplace_kernel(
    T* data,
    const int64_t numel,
    const float keep_prob,
    const float scale,
    const uint64_t seed,
    const uint64_t offset
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    
    for (int64_t i = idx; i < numel; i += stride) {
        uint64_t hash_input = seed ^ (offset + i);
        float rand_val = hash_to_float(hash_input);
        data[i] = (rand_val < keep_prob) ? static_cast<T>(data[i] * scale) : static_cast<T>(0);
    }
}

template <typename T, int VEC_SIZE>
__global__ void __launch_bounds__(256, 8) dropout_vectorized_kernel(
    T* output,
    const T* input,
    const int64_t numel,
    const float keep_prob,
    const float scale,
    const uint64_t seed,
    const uint64_t offset
) {
    using VecT = typename std::conditional<VEC_SIZE == 4, float4,
                 typename std::conditional<VEC_SIZE == 2, float2, float>::type>::type;
    
    const int64_t vec_idx = (blockIdx.x * blockDim.x + threadIdx.x);
    const int64_t idx = vec_idx * VEC_SIZE;
    
    if (idx + VEC_SIZE <= numel) {
        VecT vec_data = *reinterpret_cast<const VecT*>(&input[idx]);
        T* elem_ptr = reinterpret_cast<T*>(&vec_data);
        
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            uint64_t hash_input = seed ^ (offset + idx + i);
            float rand_val = hash_to_float(hash_input);
            elem_ptr[i] = (rand_val < keep_prob) ? static_cast<T>(elem_ptr[i] * scale) : static_cast<T>(0);
        }
        
        *reinterpret_cast<VecT*>(&output[idx]) = vec_data;
    } else if (idx < numel) {
        #pragma unroll
        for (int i = 0; i < VEC_SIZE && (idx + i) < numel; ++i) {
            uint64_t hash_input = seed ^ (offset + idx + i);
            float rand_val = hash_to_float(hash_input);
            output[idx + i] = (rand_val < keep_prob) ? static_cast<T>(input[idx + i] * scale) : static_cast<T>(0);
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
        return inplace ? input : input.clone();
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
    const int vec_size = 4;
    const int64_t vec_numel = (numel + vec_size - 1) / vec_size;
    const int blocks = min((vec_numel + threads - 1) / threads, (int64_t)82 * 8);
    
    if (inplace) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dropout_inplace", ([&] {
            if (numel >= vec_size && reinterpret_cast<uintptr_t>(input.data_ptr<scalar_t>()) % (sizeof(scalar_t) * vec_size) == 0) {
                dropout_inplace_vectorized_kernel<scalar_t, 4><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(),
                    numel,
                    keep_prob,
                    scale,
                    seed,
                    offset
                );
            } else {
                dropout_inplace_kernel<scalar_t><<<blocks, threads>>>(
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
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dropout", ([&] {
            if (numel >= vec_size && 
                reinterpret_cast<uintptr_t>(input.data_ptr<scalar_t>()) % (sizeof(scalar_t) * vec_size) == 0 &&
                reinterpret_cast<uintptr_t>(output.data_ptr<scalar_t>()) % (sizeof(scalar_t) * vec_size) == 0) {
                dropout_vectorized_kernel<scalar_t, 4><<<blocks, threads>>>(
                    output.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    numel,
                    keep_prob,
                    scale,
                    seed,
                    offset
                );
            } else {
                dropout_inplace_kernel<scalar_t><<<blocks, threads>>>(
                    output.data_ptr<scalar_t>(),
                    numel,
                    keep_prob,
                    scale,
                    seed,
                    offset
                );
                cudaMemcpy(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), numel * sizeof(scalar_t), cudaMemcpyDeviceToDevice);
            }
        }));
        
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
        
        return output;
    }
}