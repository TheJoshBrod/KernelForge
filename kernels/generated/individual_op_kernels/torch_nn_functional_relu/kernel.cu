// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ============ DEVICE CODE (CUDA kernels only) ============
template <typename T>
__global__ void relu_kernel(T* output, const T* input, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        T val = input[idx];
        output[idx] = val > T(0) ? val : T(0);
    }
}

template <typename T>
__global__ void relu_inplace_kernel(T* data, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        T val = data[idx];
        data[idx] = val > T(0) ? val : T(0);
    }
}

// ============ HOST CODE ============

torch::Tensor launch(torch::Tensor input, bool inplace) {
    // 1. Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    
    // Make contiguous if needed
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    
    // 2. Output tensor creation
    torch::Tensor output;
    if (inplace) {
        output = input;
    } else {
        output = torch::empty_like(input);
    }
    
    // 3. Get number of elements
    int64_t numel = input.numel();
    
    // 4. Kernel launch parameters
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    // 5. Launch kernel with dtype dispatch
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "relu_cuda", [&] {
        if (inplace) {
            relu_inplace_kernel<scalar_t><<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                numel
            );
        } else {
            relu_kernel<scalar_t><<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                numel
            );
        }
    });
    
    // 6. Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    // 7. Return tensor
    return output;
}

// [END kernel.cu]