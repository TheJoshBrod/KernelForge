#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <cmath>

// ============ DEVICE CODE (CUDA kernels only) ============

template <typename T>
__device__ __forceinline__ T gelu_forward(T x) {
    // GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function of the standard normal distribution
    // Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    const T sqrt_2_over_pi = T(0.7978845608028654);  // sqrt(2/pi)
    const T coeff = T(0.044715);
    const T half = T(0.5);
    const T one = T(1.0);
    
    T x_cubed = x * x * x;
    T inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    T tanh_inner = tanh(inner);
    return half * x * (one + tanh_inner);
}

// Specialization for half precision
__device__ __forceinline__ __half gelu_forward(__half x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    const float half = 0.5f;
    const float one = 1.0f;
    
    float x_float = __half2float(x);
    float x_cubed = x_float * x_float * x_float;
    float inner = sqrt_2_over_pi * (x_float + coeff * x_cubed);
    float tanh_inner = tanhf(inner);
    float result = half * x_float * (one + tanh_inner);
    return __float2half(result);
}

template <typename T>
__global__ void gelu_kernel(T* output, const T* input, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numel) {
        output[idx] = gelu_forward(input[idx]);
    }
}

// ============ HOST CODE ============

torch::Tensor launch(torch::Tensor input) {
    // 1. Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    
    // Make contiguous if needed
    auto input_contiguous = input.contiguous();
    
    // 2. Output tensor creation
    auto output = torch::empty_like(input_contiguous);
    
    // 3. Get total number of elements
    int64_t numel = input_contiguous.numel();
    
    if (numel == 0) {
        return output;
    }
    
    // 4. Kernel launch parameters
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    
    // 5. Launch kernel with dtype dispatch
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_contiguous.scalar_type(), "gelu_kernel", ([&] {
        gelu_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input_contiguous.data_ptr<scalar_t>(),
            numel
        );
    }));
    
    // 6. Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "GELU kernel failed: ", cudaGetErrorString(err));
    
    // 7. Return tensor
    return output;
}

// [END kernel.cu]