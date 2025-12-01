#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      AT_ERROR("CUDA error: ", cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

// ============ DEVICE CODE (CUDA kernels only) ============

template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T block_reduce_max(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warp_reduce_max(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -INFINITY;
    if (wid == 0) val = warp_reduce_max(val);
    
    return val;
}

template <typename T>
__device__ __forceinline__ T block_reduce_sum(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

// Optimized softmax kernel for the last dimension
template <typename T>
__global__ void softmax_kernel_last_dim(
    T* __restrict__ output,
    const T* __restrict__ input,
    int64_t outer_size,
    int64_t softmax_size
) {
    int64_t idx = blockIdx.x;
    if (idx >= outer_size) return;
    
    const T* input_row = input + idx * softmax_size;
    T* output_row = output + idx * softmax_size;
    
    // Find max value for numerical stability
    T max_val = -INFINITY;
    for (int64_t i = threadIdx.x; i < softmax_size; i += blockDim.x) {
        max_val = max(max_val, input_row[i]);
    }
    max_val = block_reduce_max(max_val);
    __shared__ T shared_max;
    if (threadIdx.x == 0) shared_max = max_val;
    __syncthreads();
    max_val = shared_max;
    
    // Compute exp(x - max) and sum
    T sum_val = 0;
    for (int64_t i = threadIdx.x; i < softmax_size; i += blockDim.x) {
        T exp_val = exp(input_row[i] - max_val);
        output_row[i] = exp_val;
        sum_val += exp_val;
    }
    sum_val = block_reduce_sum(sum_val);
    __shared__ T shared_sum;
    if (threadIdx.x == 0) shared_sum = sum_val;
    __syncthreads();
    sum_val = shared_sum;
    
    // Normalize
    for (int64_t i = threadIdx.x; i < softmax_size; i += blockDim.x) {
        output_row[i] = output_row[i] / sum_val;
    }
}

// General softmax kernel for arbitrary dimensions
template <typename T>
__global__ void softmax_kernel_general(
    T* __restrict__ output,
    const T* __restrict__ input,
    int64_t outer_size,
    int64_t softmax_size,
    int64_t inner_size
) {
    int64_t outer_idx = blockIdx.x;
    int64_t inner_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    int64_t base_idx = outer_idx * softmax_size * inner_size + inner_idx;
    
    // Find max value for numerical stability
    T max_val = -INFINITY;
    for (int64_t i = threadIdx.x; i < softmax_size; i += blockDim.x) {
        int64_t idx = base_idx + i * inner_size;
        max_val = max(max_val, input[idx]);
    }
    max_val = warp_reduce_max(max_val);
    __shared__ T shared_max[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    if (lane == 0) shared_max[wid * blockDim.y + threadIdx.y] = max_val;
    __syncthreads();
    if (wid == 0 && threadIdx.x < blockDim.x / 32) {
        max_val = shared_max[threadIdx.x * blockDim.y + threadIdx.y];
        max_val = warp_reduce_max(max_val);
        shared_max[threadIdx.y] = max_val;
    }
    __syncthreads();
    max_val = shared_max[threadIdx.y];
    
    // Compute exp(x - max) and sum
    T sum_val = 0;
    for (int64_t i = threadIdx.x; i < softmax_size; i += blockDim.x) {
        int64_t idx = base_idx + i * inner_size;
        T exp_val = exp(input[idx] - max_val);
        output[idx] = exp_val;
        sum_val += exp_val;
    }
    sum_val = warp_reduce_sum(sum_val);
    if (lane == 0) shared_max[wid * blockDim.y + threadIdx.y] = sum_val;
    __syncthreads();
    if (wid == 0 && threadIdx.x < blockDim.x / 32) {
        sum_val = shared_max[threadIdx.x * blockDim.y + threadIdx.y];
        sum_val = warp_reduce_sum(sum_val);
        shared_max[threadIdx.y] = sum_val;
    }
    __syncthreads();
    sum_val = shared_max[threadIdx.y];
    
    // Normalize
    for (int64_t i = threadIdx.x; i < softmax_size; i += blockDim.x) {
        int64_t idx = base_idx + i * inner_size;
        output[idx] = output[idx] / sum_val;
    }
}

// ============ HOST CODE ============

torch::Tensor launch(torch::Tensor input, int64_t dim, int64_t _stacklevel, c10::optional<at::ScalarType> dtype) {
    // 1. Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    
    // Handle negative dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");
    
    // 2. Output tensor creation
    auto output = torch::empty_like(input);
    
    // 3. Calculate dimensions
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }
    
    int64_t softmax_size = input.size(dim);
    
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; i++) {
        inner_size *= input.size(i);
    }
    
    // 4. Launch kernel based on dimension configuration
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "softmax_cuda", [&] {
        if (inner_size == 1) {
            // Optimized path for last dimension softmax
            int threads = min(1024, (int)((softmax_size + 31) / 32 * 32));
            threads = max(32, threads);
            int blocks = outer_size;
            
            softmax_kernel_last_dim<scalar_t><<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                outer_size,
                softmax_size
            );
        } else {
            // General path for arbitrary dimension
            int threads_x = min(256, (int)((softmax_size + 31) / 32 * 32));
            threads_x = max(32, threads_x);
            int threads_y = min(4, (int)inner_size);
            
            dim3 threads(threads_x, threads_y);
            dim3 blocks(outer_size, (inner_size + threads_y - 1) / threads_y);
            
            softmax_kernel_general<scalar_t><<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                outer_size,
                softmax_size,
                inner_size
            );
        }
    });
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 5. Return tensor
    return output;
}

// [END kernel.cu]