#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

// ============ DEVICE CODE (CUDA kernels only) ============

template <typename scalar_t>
__global__ void softmax_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size
) {
    int64_t outer_idx = blockIdx.x;
    int64_t inner_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) {
        return;
    }
    
    int64_t offset = outer_idx * dim_size * inner_size + inner_idx;
    
    // Find max value for numerical stability
    scalar_t max_val = input[offset];
    for (int64_t d = 1; d < dim_size; ++d) {
        scalar_t val = input[offset + d * inner_size];
        max_val = max(max_val, val);
    }
    
    // Compute exp(x - max) and sum
    scalar_t sum = 0;
    for (int64_t d = 0; d < dim_size; ++d) {
        int64_t idx = offset + d * inner_size;
        scalar_t exp_val = exp(input[idx] - max_val);
        output[idx] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    for (int64_t d = 0; d < dim_size; ++d) {
        int64_t idx = offset + d * inner_size;
        output[idx] /= sum;
    }
}

// Optimized kernel for last dimension (inner_size == 1)
template <typename scalar_t>
__global__ void softmax_last_dim_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= outer_size) {
        return;
    }
    
    int64_t offset = idx * dim_size;
    
    // Find max value for numerical stability
    scalar_t max_val = input[offset];
    for (int64_t d = 1; d < dim_size; ++d) {
        max_val = max(max_val, input[offset + d]);
    }
    
    // Compute exp(x - max) and sum
    scalar_t sum = 0;
    for (int64_t d = 0; d < dim_size; ++d) {
        scalar_t exp_val = exp(input[offset + d] - max_val);
        output[offset + d] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    for (int64_t d = 0; d < dim_size; ++d) {
        output[offset + d] /= sum;
    }
}

// ============ HOST CODE ============

torch::Tensor launch(torch::Tensor input, int64_t dim, int64_t _stacklevel, c10::optional<c10::ScalarType> dtype) {
    // Note: _stacklevel and dtype are ignored as they're not needed for CUDA kernel execution
    
    // 1. Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    
    // Make contiguous if needed
    auto input_contig = input.contiguous();
    
    // Handle negative dimension
    if (dim < 0) {
        dim += input_contig.dim();
    }
    
    TORCH_CHECK(dim >= 0 && dim < input_contig.dim(), 
                "dim must be in range [0, ", input_contig.dim(), ")");
    
    // 2. Output tensor creation
    auto output = torch::empty_like(input_contig);
    
    // 3. Calculate sizes
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= input_contig.size(i);
    }
    
    int64_t dim_size = input_contig.size(dim);
    
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < input_contig.dim(); ++i) {
        inner_size *= input_contig.size(i);
    }
    
    // 4. Launch kernel
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input_contig));
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_contig.scalar_type(), "softmax_kernel", [&] {
        if (inner_size == 1) {
            // Optimized path for last dimension
            int threads = 256;
            int blocks = (outer_size + threads - 1) / threads;
            
            softmax_last_dim_kernel<scalar_t><<<blocks, threads>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                outer_size,
                dim_size
            );
        } else {
            // General case
            dim3 threads(1, 256);
            dim3 blocks(outer_size, (inner_size + threads.y - 1) / threads.y);
            
            softmax_kernel<scalar_t><<<blocks, threads>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                outer_size,
                dim_size,
                inner_size
            );
        }
    });
    
    // 5. Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    // 6. Return tensor
    return output;
}

// [END kernel.cu]