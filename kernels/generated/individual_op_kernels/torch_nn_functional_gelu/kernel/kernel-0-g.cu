// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <cmath>

// ============ DEVICE CODE (CUDA kernels only) ============

// GELU activation function implementation
// GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function of the standard normal distribution
// Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

template <typename T>
__device__ __forceinline__ T gelu_impl(T x) {
    const T sqrt_2_over_pi = 0.7978845608028654;  // sqrt(2/pi)
    const T coeff = 0.044715;
    T x_cubed = x * x * x;
    T tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
    return static_cast<T>(0.5) * x * (static_cast<T>(1.0) + tanh(tanh_arg));
}

// Specialization for half precision
__device__ __forceinline__ __half gelu_impl(__half x) {
    float xf = __half2float(x);
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x_cubed = xf * xf * xf;
    float tanh_arg = sqrt_2_over_pi * (xf + coeff * x_cubed);
    float result = 0.5f * xf * (1.0f + tanhf(tanh_arg));
    return __float2half(result);
}

template <typename T>
__global__ void gelu_kernel(T* output, const T* input, int64_t total_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    for (int64_t i = idx; i < total_elements; i += stride) {
        output[i] = gelu_impl(input[i]);
    }
}

// ============ HOST CODE ============

torch::Tensor launch(torch::Tensor input) {
    // 1. Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    
    // Make input contiguous if needed
    auto input_contig = input.contiguous();
    
    // 2. Output tensor creation
    auto output = torch::empty_like(input_contig);
    
    // 3. Calculate total number of elements
    int64_t total_elements = input_contig.numel();
    
    if (total_elements == 0) {
        return output;
    }
    
    // 4. Kernel launch parameters
    const int threads = 256;
    const int blocks = std::min((total_elements + threads - 1) / threads, (int64_t)65535);
    
    // 5. Launch kernel with dtype dispatch
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_contig.scalar_type(), "gelu_kernel", ([&] {
        gelu_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input_contig.data_ptr<scalar_t>(),
            total_elements
        );
    }));
    
    // 6. Error checking
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    
    // 7. Return tensor
    return output;
}

// [END kernel.cu]