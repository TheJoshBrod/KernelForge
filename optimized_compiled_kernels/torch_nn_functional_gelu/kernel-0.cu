#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// ============ DEVICE CODE (CUDA kernels only) ============

// GELU activation function with optimized implementation
// GELU(x) = x * Phi(x) where Phi(x) is the CDF of standard normal distribution
// Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

template <typename T>
__device__ __forceinline__ T gelu_forward(T x) {
    const T sqrt_2_over_pi = T(0.7978845608028654);
    const T coeff = T(0.044715);
    const T one = T(1.0);
    const T half = T(0.5);
    
    T x_sq = x * x;
    T x_cubed = x_sq * x;
    T inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    T tanh_inner = tanh(inner);
    
    return half * x * (one + tanh_inner);
}

__device__ __forceinline__ __half gelu_forward(__half x) {
    float x_float = __half2float(x);
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    
    float x_sq = x_float * x_float;
    float x_cubed = x_sq * x_float;
    float inner = sqrt_2_over_pi * __fmaf_rn(coeff, x_cubed, x_float);
    float tanh_inner = tanhf(inner);
    
    return __float2half(__fmaf_rn(0.5f * x_float, tanh_inner, 0.5f * x_float));
}

template <typename T>
__global__ void gelu_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    const int64_t total_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    #pragma unroll 4
    for (int64_t i = idx; i < total_elements; i += stride) {
        T val = input[i];
        output[i] = gelu_forward(val);
    }
}

template <typename T>
__global__ void gelu_kernel_vec4(
    T* __restrict__ output,
    const T* __restrict__ input,
    const int64_t total_elements
) {
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int64_t stride = blockDim.x * gridDim.x * 4;
    
    for (int64_t i = idx; i < total_elements - 3; i += stride) {
        T val0 = input[i];
        T val1 = input[i + 1];
        T val2 = input[i + 2];
        T val3 = input[i + 3];
        
        output[i] = gelu_forward(val0);
        output[i + 1] = gelu_forward(val1);
        output[i + 2] = gelu_forward(val2);
        output[i + 3] = gelu_forward(val3);
    }
}

// ============ HOST CODE ============

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
    const int sm_count = 82;
    const int blocks_per_sm = 4;
    const int max_blocks = sm_count * blocks_per_sm;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_contig.scalar_type(), "gelu_cuda", [&] {
        if (total_elements >= 1024 && total_elements % 4 == 0) {
            int64_t vec_elements = total_elements / 4;
            int blocks = std::min((int)((vec_elements + threads - 1) / threads), max_blocks);
            
            gelu_kernel_vec4<scalar_t><<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                input_contig.data_ptr<scalar_t>(),
                total_elements
            );
            
            int64_t remainder_start = (vec_elements * 4);
            if (remainder_start < total_elements) {
                gelu_kernel<scalar_t><<<1, 256>>>(
                    output.data_ptr<scalar_t>() + remainder_start,
                    input_contig.data_ptr<scalar_t>() + remainder_start,
                    total_elements - remainder_start
                );
            }
        } else {
            int blocks = std::min((int)((total_elements + threads - 1) / threads), max_blocks);
            gelu_kernel<scalar_t><<<blocks, threads>>>(
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