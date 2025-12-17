// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

// ============ DEVICE CODE (CUDA kernels only) ============

template <typename T>
__global__ void layer_norm_kernel(
    T* output,
    const T* input,
    const T* weight,
    const T* bias,
    const float* mean,
    const float* rstd,
    int batch_size,
    int seq_len,
    int normalized_size,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * normalized_size;
    
    if (idx < total_elements) {
        int norm_idx = idx % normalized_size;
        int batch_seq_idx = idx / normalized_size;
        
        float m = mean[batch_seq_idx];
        float r = rstd[batch_seq_idx];
        
        float normalized = (static_cast<float>(input[idx]) - m) * r;
        
        float w = weight ? static_cast<float>(weight[norm_idx]) : 1.0f;
        float b = bias ? static_cast<float>(bias[norm_idx]) : 0.0f;
        
        output[idx] = static_cast<T>(normalized * w + b);
    }
}

template <typename T>
__global__ void compute_mean_kernel(
    float* mean,
    const T* input,
    int batch_size,
    int seq_len,
    int normalized_size
) {
    int batch_seq_idx = blockIdx.x;
    if (batch_seq_idx >= batch_size * seq_len) return;
    
    float sum = 0.0f;
    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        int idx = batch_seq_idx * normalized_size + i;
        sum += static_cast<float>(input[idx]);
    }
    
    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Block reduction
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    if (lane == 0) shared[wid] = sum;
    __syncthreads();
    
    if (wid == 0) {
        sum = (threadIdx.x < (blockDim.x / warpSize)) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (threadIdx.x == 0) {
            mean[batch_seq_idx] = sum / normalized_size;
        }
    }
}

template <typename T>
__global__ void compute_rstd_kernel(
    float* rstd,
    const float* mean,
    const T* input,
    int batch_size,
    int seq_len,
    int normalized_size,
    float eps
) {
    int batch_seq_idx = blockIdx.x;
    if (batch_seq_idx >= batch_size * seq_len) return;
    
    float m = mean[batch_seq_idx];
    float var_sum = 0.0f;
    
    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        int idx = batch_seq_idx * normalized_size + i;
        float diff = static_cast<float>(input[idx]) - m;
        var_sum += diff * diff;
    }
    
    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }
    
    // Block reduction
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    if (lane == 0) shared[wid] = var_sum;
    __syncthreads();
    
    if (wid == 0) {
        var_sum = (threadIdx.x < (blockDim.x / warpSize)) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
        }
        if (threadIdx.x == 0) {
            float variance = var_sum / normalized_size;
            rstd[batch_seq_idx] = rsqrtf(variance + eps);
        }
    }
}

// ============ HOST CODE ============

torch::Tensor launch(
    torch::Tensor input,
    c10::IntArrayRef normalized_shape,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    // Make input contiguous if needed
    input = input.contiguous();
    
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    
    if (weight.defined()) {
        weight = weight.contiguous();
        TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    }
    
    if (bias.defined()) {
        bias = bias.contiguous();
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    }
    
    // Get dimensions
    auto input_sizes = input.sizes();
    int64_t batch_size = input_sizes[0];
    int64_t seq_len = input_sizes[1];
    int64_t normalized_size = normalized_shape[0];
    
    TORCH_CHECK(input_sizes[2] == normalized_size, 
                "Last dimension of input must match normalized_shape");
    
    // Create output tensor
    auto output = torch::empty_like(input);
    
    // Create temporary buffers for mean and rstd
    auto mean = torch::empty({batch_size * seq_len}, 
                            torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
    auto rstd = torch::empty({batch_size * seq_len}, 
                            torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
    
    int batch_seq_total = batch_size * seq_len;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "layer_norm_cuda", [&] {
        // Step 1: Compute mean
        int threads_mean = 256;
        compute_mean_kernel<scalar_t><<<batch_seq_total, threads_mean>>>(
            mean.data_ptr<float>(),
            input.data_ptr<scalar_t>(),
            batch_size,
            seq_len,
            normalized_size
        );
        
        // Step 2: Compute rstd (1/sqrt(variance + eps))
        int threads_rstd = 256;
        compute_rstd_kernel<scalar_t><<<batch_seq_total, threads_rstd>>>(
            rstd.data_ptr<float>(),
            mean.data_ptr<float>(),
            input.data_ptr<scalar_t>(),
            batch_size,
            seq_len,
            normalized_size,
            static_cast<float>(eps)
        );
        
        // Step 3: Normalize and apply affine transformation
        int total_elements = batch_size * seq_len * normalized_size;
        int threads_norm = 256;
        int blocks_norm = (total_elements + threads_norm - 1) / threads_norm;
        
        layer_norm_kernel<scalar_t><<<blocks_norm, threads_norm>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.defined() ? weight.data_ptr<scalar_t>() : nullptr,
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            mean.data_ptr<float>(),
            rstd.data_ptr<float>(),
            batch_size,
            seq_len,
            normalized_size,
            static_cast<float>(eps)
        );
    });
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return output;
}

// [END kernel.cu]