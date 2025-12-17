// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ============ DEVICE CODE (CUDA kernels only) ============

template <typename scalar_t>
__global__ void batch_norm_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ running_mean,
    const scalar_t* __restrict__ running_var,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int N, int C, int spatial_size,
    double eps,
    bool training,
    double momentum) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = N * C * spatial_size;
    
    if (idx < total_size) {
        int spatial_idx = idx % spatial_size;
        int c = (idx / spatial_size) % C;
        int n = idx / (C * spatial_size);
        
        scalar_t mean_val = running_mean[c];
        scalar_t var_val = running_var[c];
        scalar_t weight_val = weight ? weight[c] : scalar_t(1.0);
        scalar_t bias_val = bias ? bias[c] : scalar_t(0.0);
        
        // Normalize: (x - mean) / sqrt(var + eps)
        scalar_t normalized = (input[idx] - mean_val) / sqrt(var_val + scalar_t(eps));
        
        // Scale and shift: gamma * normalized + beta
        output[idx] = weight_val * normalized + bias_val;
    }
}

template <typename scalar_t>
__global__ void compute_mean_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ mean,
    int N, int C, int spatial_size) {
    
    int c = blockIdx.x;
    if (c >= C) return;
    
    // Each block computes mean for one channel
    __shared__ scalar_t shared_sum[256];
    
    int tid = threadIdx.x;
    int elements_per_channel = N * spatial_size;
    
    scalar_t local_sum = 0;
    for (int i = tid; i < elements_per_channel; i += blockDim.x) {
        int n = i / spatial_size;
        int spatial_idx = i % spatial_size;
        int idx = n * C * spatial_size + c * spatial_size + spatial_idx;
        local_sum += input[idx];
    }
    
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        mean[c] = shared_sum[0] / scalar_t(elements_per_channel);
    }
}

template <typename scalar_t>
__global__ void compute_var_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ mean,
    scalar_t* __restrict__ var,
    int N, int C, int spatial_size) {
    
    int c = blockIdx.x;
    if (c >= C) return;
    
    __shared__ scalar_t shared_sum[256];
    
    int tid = threadIdx.x;
    int elements_per_channel = N * spatial_size;
    scalar_t mean_val = mean[c];
    
    scalar_t local_sum = 0;
    for (int i = tid; i < elements_per_channel; i += blockDim.x) {
        int n = i / spatial_size;
        int spatial_idx = i % spatial_size;
        int idx = n * C * spatial_size + c * spatial_size + spatial_idx;
        scalar_t diff = input[idx] - mean_val;
        local_sum += diff * diff;
    }
    
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        var[c] = shared_sum[0] / scalar_t(elements_per_channel);
    }
}

template <typename scalar_t>
__global__ void update_running_stats_kernel(
    scalar_t* __restrict__ running_mean,
    scalar_t* __restrict__ running_var,
    const scalar_t* __restrict__ batch_mean,
    const scalar_t* __restrict__ batch_var,
    int C,
    double momentum) {
    
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C) {
        scalar_t mom = scalar_t(momentum);
        running_mean[c] = (scalar_t(1.0) - mom) * running_mean[c] + mom * batch_mean[c];
        running_var[c] = (scalar_t(1.0) - mom) * running_var[c] + mom * batch_var[c];
    }
}

// ============ HOST CODE ============

torch::Tensor launch(
    torch::Tensor arg0,
    torch::Tensor arg1,
    torch::Tensor arg2,
    torch::Tensor arg3,
    torch::Tensor arg4,
    bool arg5,
    double arg6,
    double arg7) {
    
    // arg0: input [N, C, H, W]
    // arg1: running_mean [C]
    // arg2: running_var [C]
    // arg3: weight (gamma) [C]
    // arg4: bias (beta) [C]
    // arg5: training
    // arg6: momentum
    // arg7: eps
    
    TORCH_CHECK(arg0.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(arg0.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(arg1.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(arg2.is_cuda(), "running_var must be a CUDA tensor");
    
    auto input = arg0;
    auto running_mean = arg1;
    auto running_var = arg2;
    auto weight = arg3;
    auto bias = arg4;
    bool training = arg5;
    double momentum = arg6;
    double eps = arg7;
    
    int N = input.size(0);
    int C = input.size(1);
    int spatial_size = 1;
    for (int i = 2; i < input.dim(); i++) {
        spatial_size *= input.size(i);
    }
    
    auto output = torch::empty_like(input);
    
    // If training, compute batch statistics
    if (training) {
        auto batch_mean = torch::empty({C}, input.options());
        auto batch_var = torch::empty({C}, input.options());
        
        int threads = 256;
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "batch_norm_cuda", ([&] {
            // Compute mean
            compute_mean_kernel<scalar_t><<<C, threads>>>(
                input.data_ptr<scalar_t>(),
                batch_mean.data_ptr<scalar_t>(),
                N, C, spatial_size
            );
            
            // Compute variance
            compute_var_kernel<scalar_t><<<C, threads>>>(
                input.data_ptr<scalar_t>(),
                batch_mean.data_ptr<scalar_t>(),
                batch_var.data_ptr<scalar_t>(),
                N, C, spatial_size
            );
            
            // Update running statistics
            int blocks_stats = (C + threads - 1) / threads;
            update_running_stats_kernel<scalar_t><<<blocks_stats, threads>>>(
                running_mean.data_ptr<scalar_t>(),
                running_var.data_ptr<scalar_t>(),
                batch_mean.data_ptr<scalar_t>(),
                batch_var.data_ptr<scalar_t>(),
                C,
                momentum
            );
            
            // Apply normalization using batch statistics
            int total_size = N * C * spatial_size;
            int blocks = (total_size + threads - 1) / threads;
            
            batch_norm_forward_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                batch_mean.data_ptr<scalar_t>(),
                batch_var.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<scalar_t>() : nullptr,
                bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
                output.data_ptr<scalar_t>(),
                N, C, spatial_size,
                eps,
                training,
                momentum
            );
        }));
    } else {
        // Use running statistics
        int threads = 256;
        int total_size = N * C * spatial_size;
        int blocks = (total_size + threads - 1) / threads;
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "batch_norm_cuda", ([&] {
            batch_norm_forward_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                running_mean.data_ptr<scalar_t>(),
                running_var.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<scalar_t>() : nullptr,
                bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
                output.data_ptr<scalar_t>(),
                N, C, spatial_size,
                eps,
                training,
                momentum
            );
        }));
    }
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

// [END kernel.cu]