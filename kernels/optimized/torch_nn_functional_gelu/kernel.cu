#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

template <typename T>
__device__ __forceinline__ T gelu_forward(T x) {
    const T sqrt_2_over_pi = T(0.7978845608028654);
    const T coeff = T(0.044715);
    const T one = T(1.0);
    const T half = T(0.5);
    
    T x_cubed = x * x * x;
    T inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    T tanh_inner = tanh(inner);
    
    return half * x * (one + tanh_inner);
}

__device__ __forceinline__ __half gelu_forward(__half x) {
    float x_float = __half2float(x);
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    
    float x_cubed = x_float * x_float * x_float;
    float inner = sqrt_2_over_pi * (x_float + coeff * x_cubed);
    float tanh_inner = tanhf(inner);
    
    return __float2half(0.5f * x_float * (1.0f + tanh_inner));
}

template <typename T>
__global__ void gelu_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    const int64_t total_elements
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    
    for (int64_t i = idx; i < total_elements; i += stride) {
        T val = input[i];
        output[i] = gelu_forward(val);
    }
}

template <typename T, int VEC_SIZE>
__global__ void gelu_kernel_vectorized(
    T* __restrict__ output,
    const T* __restrict__ input,
    const int64_t total_elements
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t vec_idx = idx * VEC_SIZE;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x * VEC_SIZE;
    
    for (int64_t i = vec_idx; i < total_elements; i += stride) {
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            if (i + j < total_elements) {
                T val = input[i + j];
                output[i + j] = gelu_forward(val);
            }
        }
    }
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

torch::Tensor launch(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    
    auto input_contig = input.contiguous();
    int64_t total_elements = input_contig.numel();
    auto output = torch::empty_like(input_contig);
    
    const int threads = 256;
    const int max_blocks = 65535;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_contig.scalar_type(), "gelu_cuda", [&] {
        constexpr int VEC_SIZE = 4;
        bool use_vectorized = (total_elements >= VEC_SIZE * threads) && 
                              (reinterpret_cast<uintptr_t>(input_contig.data_ptr<scalar_t>()) % (VEC_SIZE * sizeof(scalar_t)) == 0) &&
                              (reinterpret_cast<uintptr_t>(output.data_ptr<scalar_t>()) % (VEC_SIZE * sizeof(scalar_t)) == 0);
        
        if (use_vectorized) {
            int64_t vec_elements = (total_elements + VEC_SIZE - 1) / VEC_SIZE;
            int blocks = (vec_elements + threads - 1) / threads;
            int final_blocks = std::min(blocks, max_blocks);
            
            gelu_kernel_vectorized<scalar_t, VEC_SIZE><<<final_blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                input_contig.data_ptr<scalar_t>(),
                total_elements
            );
        } else {
            int blocks = (total_elements + threads - 1) / threads;
            int final_blocks = std::min(blocks, max_blocks);
            
            gelu_kernel<scalar_t><<<final_blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                input_contig.data_ptr<scalar_t>(),
                total_elements
            );
        }
    });
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return output;
}