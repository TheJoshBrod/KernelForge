// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

// ============ DEVICE CODE (CUDA kernels only) ============

template <typename scalar_t>
__global__ void linear_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int64_t batch_size,
    const int64_t in_features,
    const int64_t out_features,
    const bool has_bias
) {
    // Each thread computes one output element
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_elements = batch_size * out_features;
    
    if (idx < total_elements) {
        const int64_t batch_idx = idx / out_features;
        const int64_t out_idx = idx % out_features;
        
        scalar_t sum = 0;
        
        // Compute dot product: input[batch_idx, :] @ weight[out_idx, :]
        for (int64_t k = 0; k < in_features; ++k) {
            sum += input[batch_idx * in_features + k] * weight[out_idx * in_features + k];
        }
        
        // Add bias if present
        if (has_bias) {
            sum += bias[out_idx];
        }
        
        output[idx] = sum;
    }
}

// Optimized kernel using shared memory and tiling
template <typename scalar_t>
__global__ void linear_kernel_tiled(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int64_t batch_size,
    const int64_t in_features,
    const int64_t out_features,
    const bool has_bias
) {
    const int TILE_SIZE = 32;
    __shared__ scalar_t s_input[32][32];
    __shared__ scalar_t s_weight[32][32];
    
    const int64_t row = blockIdx.y * blockDim.y + threadIdx.y; // batch index
    const int64_t col = blockIdx.x * blockDim.x + threadIdx.x; // output feature index
    
    scalar_t sum = 0;
    
    // Tile the computation
    for (int64_t tile = 0; tile < (in_features + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load input tile
        int64_t in_col = tile * TILE_SIZE + threadIdx.x;
        if (row < batch_size && in_col < in_features) {
            s_input[threadIdx.y][threadIdx.x] = input[row * in_features + in_col];
        } else {
            s_input[threadIdx.y][threadIdx.x] = 0;
        }
        
        // Load weight tile (transposed: weight[col, in_col])
        int64_t in_row = tile * TILE_SIZE + threadIdx.y;
        if (col < out_features && in_row < in_features) {
            s_weight[threadIdx.y][threadIdx.x] = weight[col * in_features + in_row];
        } else {
            s_weight[threadIdx.y][threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_input[threadIdx.y][k] * s_weight[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < batch_size && col < out_features) {
        if (has_bias) {
            sum += bias[col];
        }
        output[row * out_features + col] = sum;
    }
}

// ============ HOST CODE ============

torch::Tensor launch(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias
) {
    // 1. Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.value().is_contiguous(), "bias must be contiguous");
    }
    
    // 2. Get dimensions
    // Input shape: [*, in_features] where * can be multiple dimensions
    // Weight shape: [out_features, in_features]
    // Output shape: [*, out_features]
    
    auto input_sizes = input.sizes().vec();
    const int64_t in_features = input_sizes.back();
    const int64_t out_features = weight.size(0);
    
    TORCH_CHECK(weight.size(1) == in_features, 
                "weight dimension 1 must match input last dimension");
    
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().size(0) == out_features,
                    "bias size must match weight dimension 0");
    }
    
    // Reshape input to 2D: [batch_size, in_features]
    auto input_2d = input.reshape({-1, in_features});
    const int64_t batch_size = input_2d.size(0);
    
    // 3. Create output tensor
    auto output_sizes = input_sizes;
    output_sizes.back() = out_features;
    auto output = torch::empty(output_sizes, input.options());
    auto output_2d = output.reshape({batch_size, out_features});
    
    // 4. Determine if bias is present
    const bool has_bias = bias.has_value();
    const float* bias_ptr = has_bias ? bias.value().data_ptr<float>() : nullptr;
    
    // 5. Launch kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "linear_cuda", ([&] {
        const scalar_t* input_ptr = input_2d.data_ptr<scalar_t>();
        const scalar_t* weight_ptr = weight.data_ptr<scalar_t>();
        const scalar_t* bias_ptr_typed = has_bias ? bias.value().data_ptr<scalar_t>() : nullptr;
        scalar_t* output_ptr = output_2d.data_ptr<scalar_t>();
        
        const int64_t total_elements = batch_size * out_features;
        
        // Choose kernel based on problem size
        if (batch_size >= 32 && out_features >= 32) {
            // Use tiled kernel for larger problems
            dim3 threads(32, 32);
            dim3 blocks(
                (out_features + threads.x - 1) / threads.x,
                (batch_size + threads.y - 1) / threads.y
            );
            
            linear_kernel_tiled<scalar_t><<<blocks, threads>>>(
                input_ptr,
                weight_ptr,
                bias_ptr_typed,
                output_ptr,
                batch_size,
                in_features,
                out_features,
                has_bias
            );
        } else {
            // Use simple kernel for smaller problems
            const int threads = 256;
            const int blocks = (total_elements + threads - 1) / threads;
            
            linear_kernel<scalar_t><<<blocks, threads>>>(
                input_ptr,
                weight_ptr,
                bias_ptr_typed,
                output_ptr,
                batch_size,
                in_features,
                out_features,
                has_bias
            );
        }
    }));
    
    // 6. Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    // 7. Return output with original shape
    return output;
}

// [END kernel.cu]