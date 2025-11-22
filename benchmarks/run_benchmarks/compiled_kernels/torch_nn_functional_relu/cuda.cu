#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ============ DEVICE CODE (CUDA kernels only) ============
template <typename T>
__global__ void relu_kernel(T* output, const T* input, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        T zero = static_cast<T>(0);
        output[idx] = input[idx] > zero ? input[idx] : zero;
    }
}

template <typename T>
__global__ void relu_inplace_kernel(T* data, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        T zero = static_cast<T>(0);
        if (data[idx] < zero) {
            data[idx] = zero;
        }
    }
}

// ============ HOST CODE ============

torch::Tensor launch(torch::Tensor input, bool inplace) {
    // 1. Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    
    // Make input contiguous if needed
    auto input_contig = input.contiguous();
    
    // 2. Get tensor info
    int64_t numel = input_contig.numel();
    
    // 3. Kernel launch parameters
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    
    // 4. Handle inplace vs out-of-place
    torch::Tensor output;
    
    if (inplace) {
        // Modify input tensor in-place
        output = input_contig;
        
        // 5. Launch kernel with dtype dispatch
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_contig.scalar_type(), "relu_inplace_cuda", ([&] {
            relu_inplace_kernel<scalar_t><<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                numel
            );
        }));
    } else {
        // Create output tensor
        output = torch::empty_like(input_contig);
        
        // 5. Launch kernel with dtype dispatch
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_contig.scalar_type(), "relu_cuda", ([&] {
            relu_kernel<scalar_t><<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                input_contig.data_ptr<scalar_t>(),
                numel
            );
        }));
    }
    
    // 6. Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    // 7. Return tensor
    return output;
}

// NO PYBIND11_MODULE HERE!
// [END kernel.cu]