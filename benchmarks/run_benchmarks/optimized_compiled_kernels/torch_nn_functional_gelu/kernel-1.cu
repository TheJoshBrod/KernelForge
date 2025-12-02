#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// ============ DEVICE CODE (CUDA kernels only) ============

// Optimized GELU using fast tanh approximation and vectorized loads
template <typename T>
__device__ __forceinline__ T fast_tanh(T x) {
    // Rational approximation for tanh that's faster than standard library
    T x2 = x * x;
    T numerator = x * (T(27.0) + x2);
    T denominator = T(27.0) + T(9.0) * x2;
    return numerator / denominator;
}

template <typename T>
__device__ __forceinline__ T gelu_forward_optimized(T x) {
    const T sqrt_2_over_pi = T(0.7978845608028654);
    const T coeff = T(0.044715);
    
    T x2 = x * x;
    T x3 = x2 * x;
    T inner = sqrt_2_over_pi * (x + coeff * x3);
    T tanh_inner = fast_tanh(inner);
    
    return T(0.5) * x * (T(1.0) + tanh_inner);
}

__device__ __forceinline__ __half gelu_forward_optimized(__half x) {
    float x_float = __half2float(x);
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    
    float x2 = x_float * x_float;
    float x3 = x2 * x_float;
    float inner = sqrt_2_over_pi * (x_float + coeff * x3);
    
    float x2_inner = inner * inner;
    float numerator = inner * (27.0f + x2_inner);
    float denominator = 27.0f + 9.0f * x2_inner;
    float tanh_inner = numerator / denominator;
    
    return __float2half(0.5f * x_float * (1.0f + tanh_inner));
}

// Vectorized float4 version for better memory coalescing
__global__ void gelu_kernel_float4(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int64_t total_elements
) {
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < total_elements) {
        float4 val = *reinterpret_cast<const float4*>(input + idx);
        
        val.x = gelu_forward_optimized(val.x);
        val.y = gelu_forward_optimized(val.y);
        val.z = gelu_forward_optimized(val.z);
        val.w = gelu_forward_optimized(val.w);
        
        *reinterpret_cast<float4*>(output + idx) = val;
    } else if (idx < total_elements) {
        for (int i = 0; i < 4 && idx + i < total_elements; ++i) {
            output[idx + i] = gelu_forward_optimized(input[idx + i]);
        }
    }
}

// Vectorized half2 version
__global__ void gelu_kernel_half2(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const int64_t total_elements
) {
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    if (idx + 7 < total_elements) {
        float4 val = *reinterpret_cast<const float4*>(input + idx);
        __half2* h2_ptr = reinterpret_cast<__half2*>(&val);
        
        h2_ptr[0].x = gelu_forward_optimized(h2_ptr[0].x);
        h2_ptr[0].y = gelu_forward_optimized(h2_ptr[0].y);
        h2_ptr[1].x = gelu_forward_optimized(h2_ptr[1].x);
        h2_ptr[1].y = gelu_forward_optimized(h2_ptr[1].y);
        h2_ptr[2].x = gelu_forward_optimized(h2_ptr[2].x);
        h2_ptr[2].y = gelu_forward_optimized(h2_ptr[2].y);
        h2_ptr[3].x = gelu_forward_optimized(h2_ptr[3].x);
        h2_ptr[3].y = gelu_forward_optimized(h2_ptr[3].y);
        
        *reinterpret_cast<float4*>(output + idx) = val;
    } else if (idx < total_elements) {
        for (int i = 0; i < 8 && idx + i < total_elements; ++i) {
            output[idx + i] = gelu_forward_optimized(input[idx + i]);
        }
    }
}

template <typename T>
__global__ void gelu_kernel_scalar(
    T* __restrict__ output,
    const T* __restrict__ input,
    const int64_t total_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    for (int64_t i = idx; i < total_elements; i += stride) {
        output[i] = gelu_forward_optimized(input[i]);
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
    
    if (input_contig.scalar_type() == torch::kFloat32) {
        int64_t vec_elements = total_elements / 4;
        int blocks = (vec_elements + threads - 1) / threads;
        blocks = std::min(blocks, (int64_t)65535);
        
        gelu_kernel_float4<<<blocks, threads>>>(
            output.data_ptr<float>(),
            input_contig.data_ptr<float>(),
            total_elements
        );
    } else if (input_contig.scalar_type() == torch::kFloat16) {
        int64_t vec_elements = total_elements / 8;
        int blocks = (vec_elements + threads - 1) / threads;
        blocks = std::min(blocks, (int64_t)65535);
        
        gelu_kernel_half2<<<blocks, threads>>>(
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(input_contig.data_ptr<at::Half>()),
            total_elements
        );
    } else {
        int blocks = (total_elements + threads - 1) / threads;
        blocks = std::min(blocks, (int)65535);
        
        AT_DISPATCH_FLOATING_TYPES(input_contig.scalar_type(), "gelu_cuda", [&] {
            gelu_kernel_scalar<scalar_t><<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                input_contig.data_ptr<scalar_t>(),
                total_elements
            );
        });
    }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return output;
}