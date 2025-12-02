#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename T>
__global__ void relu_kernel(T* __restrict__ output, const T* __restrict__ input, int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    
    #pragma unroll 4
    for (int64_t i = idx; i < numel; i += stride) {
        T val = input[i];
        output[i] = val > static_cast<T>(0) ? val : static_cast<T>(0);
    }
}

template <typename T>
__global__ void relu_inplace_kernel(T* __restrict__ data, int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    
    #pragma unroll 4
    for (int64_t i = idx; i < numel; i += stride) {
        T val = data[i];
        data[i] = val > static_cast<T>(0) ? val : static_cast<T>(0);
    }
}

torch::Tensor launch(torch::Tensor input, bool inplace) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    
    auto input_contig = input.contiguous();
    int64_t numel = input_contig.numel();
    
    const int threads = 256;
    const int blocks = min((numel + threads - 1) / threads, (int64_t)10240);
    
    torch::Tensor output;
    
    if (inplace) {
        output = input_contig;
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_contig.scalar_type(), "relu_inplace_cuda", ([&] {
            relu_inplace_kernel<scalar_t><<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                numel
            );
        }));
    } else {
        output = torch::empty_like(input_contig);
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_contig.scalar_type(), "relu_cuda", ([&] {
            relu_kernel<scalar_t><<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                input_contig.data_ptr<scalar_t>(),
                numel
            );
        }));
    }
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}