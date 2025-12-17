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
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    // Grid-stride loop over outer dimensions
    for (int64_t idx = blockIdx.x; idx < outer_size * inner_size; idx += gridDim.x) {
        int64_t outer_idx = idx / inner_size;
        int64_t inner_idx = idx % inner_size;
        
        // Compute base offset for this (outer, inner) pair
        int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
        
        // Find max value for numerical stability
        scalar_t max_val = input[base_offset];
        for (int64_t d = 1; d < dim_size; ++d) {
            int64_t offset = base_offset + d * inner_size;
            scalar_t val = input[offset];
            max_val = max(max_val, val);
        }
        
        // Compute exp(x - max) and sum
        scalar_t sum_exp = 0;
        for (int64_t d = 0; d < dim_size; ++d) {
            int64_t offset = base_offset + d * inner_size;
            scalar_t val = exp(input[offset] - max_val);
            output[offset] = val;
            sum_exp += val;
        }
        
        // Normalize by dividing by sum
        for (int64_t d = 0; d < dim_size; ++d) {
            int64_t offset = base_offset + d * inner_size;
            output[offset] /= sum_exp;
        }
    }
}

template <typename scalar_t>
__global__ void softmax_kernel_vectorized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    // Each block handles one (outer, inner) pair
    int64_t idx = blockIdx.x;
    if (idx >= outer_size * inner_size) return;
    
    int64_t outer_idx = idx / inner_size;
    int64_t inner_idx = idx % inner_size;
    int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
    
    // Shared memory for reduction
    extern __shared__ char shared_mem[];
    scalar_t* shared_max = reinterpret_cast<scalar_t*>(shared_mem);
    scalar_t* shared_sum = reinterpret_cast<scalar_t*>(shared_mem + blockDim.x * sizeof(scalar_t));
    
    // Phase 1: Find max value (parallel reduction)
    scalar_t thread_max = -INFINITY;
    for (int64_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
        int64_t offset = base_offset + d * inner_size;
        thread_max = max(thread_max, input[offset]);
    }
    shared_max[threadIdx.x] = thread_max;
    __syncthreads();
    
    // Reduction to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_max[threadIdx.x] = max(shared_max[threadIdx.x], shared_max[threadIdx.x + s]);
        }
        __syncthreads();
    }
    scalar_t max_val = shared_max[0];
    __syncthreads();
    
    // Phase 2: Compute exp and sum
    scalar_t thread_sum = 0;
    for (int64_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
        int64_t offset = base_offset + d * inner_size;
        scalar_t val = exp(input[offset] - max_val);
        output[offset] = val;
        thread_sum += val;
    }
    shared_sum[threadIdx.x] = thread_sum;
    __syncthreads();
    
    // Reduction to find global sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    scalar_t sum_exp = shared_sum[0];
    __syncthreads();
    
    // Phase 3: Normalize
    for (int64_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
        int64_t offset = base_offset + d * inner_size;
        output[offset] /= sum_exp;
    }
}

// ============ HOST CODE ============

torch::Tensor launch(torch::Tensor arg0, int64_t dim) {
    // Input validation
    TORCH_CHECK(arg0.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(arg0.dim() > 0, "input must have at least 1 dimension");
    
    // Handle negative dimension
    if (dim < 0) {
        dim += arg0.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < arg0.dim(), "dim out of range");
    
    // Make input contiguous
    auto input = arg0.contiguous();
    
    // Create output tensor
    auto output = torch::empty_like(input);
    
    // Calculate dimension sizes
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= input.size(i);
    }
    
    int64_t dim_size = input.size(dim);
    
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < input.dim(); ++i) {
        inner_size *= input.size(i);
    }
    
    // Set CUDA device
    c10::cuda::CUDAGuard device_guard(input.device());
    
    // Launch kernel
    int64_t total_problems = outer_size * inner_size;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "softmax_kernel", [&] {
        if (dim_size > 128 && total_problems < 65536) {
            // Use vectorized kernel for larger dim_size
            int threads = std::min(256, (int)((dim_size + 31) / 32) * 32);
            int blocks = std::min((int)total_problems, 65536);
            int shared_mem = 2 * threads * sizeof(scalar_t);
            
            softmax_kernel_vectorized<scalar_t><<<blocks, threads, shared_mem>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                outer_size,
                dim_size,
                inner_size
            );
        } else {
            // Use simple kernel for small dim_size or many problems
            int blocks = std::min((int)total_problems, 65536);
            
            softmax_kernel<scalar_t><<<blocks, 1>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                outer_size,
                dim_size,
                inner_size
            );
        }
    });
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

// [END kernel.cu]