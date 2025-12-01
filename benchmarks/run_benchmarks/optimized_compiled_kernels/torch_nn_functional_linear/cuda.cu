#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ============ DEVICE CODE (CUDA kernels only) ============

// Optimized matrix multiplication kernel for GTX 1660 Ti (Turing architecture)
// Using shared memory tiling for better cache utilization
template<typename scalar_t>
__global__ void matmul_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int64_t batch_size,
    const int64_t in_features,
    const int64_t out_features)
{
    // Tile size optimized for Turing (GTX 1660 Ti has 48KB shared memory per SM)
    const int TILE_SIZE = 32;
    
    __shared__ scalar_t tile_input[32][32];
    __shared__ scalar_t tile_weight[32][32];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    scalar_t sum = 0;
    
    // Loop over tiles
    for (int t = 0; t < (in_features + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load input tile (batch_size x in_features)
        int input_col = t * TILE_SIZE + threadIdx.x;
        if (row < batch_size && input_col < in_features) {
            tile_input[threadIdx.y][threadIdx.x] = input[row * in_features + input_col];
        } else {
            tile_input[threadIdx.y][threadIdx.x] = 0;
        }
        
        // Load weight tile transposed (out_features x in_features)
        int weight_row = t * TILE_SIZE + threadIdx.y;
        if (col < out_features && weight_row < in_features) {
            tile_weight[threadIdx.y][threadIdx.x] = weight[col * in_features + weight_row];
        } else {
            tile_weight[threadIdx.y][threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_input[threadIdx.y][k] * tile_weight[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < batch_size && col < out_features) {
        output[row * out_features + col] = sum;
    }
}

// Kernel to add bias
template<typename scalar_t>
__global__ void add_bias_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ bias,
    const int64_t batch_size,
    const int64_t out_features)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    if (idx < total_elements) {
        int col = idx % out_features;
        output[idx] += bias[col];
    }
}

// ============ HOST CODE ============

#include <torch/extension.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(status)); \
        } \
    } while(0)

torch::Tensor launch(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias)
{
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(input.dtype() == weight.dtype(), "input and weight must have same dtype");
    
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias->dtype() == input.dtype(), "bias must have same dtype as input");
    }
    
    // Get dimensions
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    TORCH_CHECK(weight_sizes.size() == 2, "weight must be 2D");
    
    int64_t out_features = weight_sizes[0];
    int64_t in_features = weight_sizes[1];
    
    // Reshape input to 2D if needed
    torch::Tensor input_2d = input;
    std::vector<int64_t> original_shape;
    int64_t batch_size = 1;
    
    if (input.dim() > 2) {
        // Save original shape for later
        for (int i = 0; i < input.dim(); ++i) {
            original_shape.push_back(input_sizes[i]);
        }
        
        // Compute batch size (product of all dims except last)
        for (int i = 0; i < input.dim() - 1; ++i) {
            batch_size *= input_sizes[i];
        }
        
        input_2d = input.reshape({batch_size, in_features});
    } else if (input.dim() == 2) {
        batch_size = input_sizes[0];
        TORCH_CHECK(input_sizes[1] == in_features, 
                   "input feature dimension must match weight");
    } else if (input.dim() == 1) {
        batch_size = 1;
        TORCH_CHECK(input_sizes[0] == in_features,
                   "input feature dimension must match weight");
        input_2d = input.unsqueeze(0);
        original_shape.push_back(input_sizes[0]);
    }
    
    input_2d = input_2d.contiguous();
    
    // Create output tensor
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    // Launch parameters
    const int TILE_SIZE = 32;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (out_features + TILE_SIZE - 1) / TILE_SIZE,
        (batch_size + TILE_SIZE - 1) / TILE_SIZE
    );
    
    // Dispatch based on dtype
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "linear_cuda", ([&] {
        matmul_kernel<scalar_t><<<blocks, threads>>>(
            input_2d.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_features,
            out_features
        );
    }));
    
    CUDA_CHECK(cudaGetLastError());
    
    // Add bias if provided
    if (bias.has_value()) {
        int threads_bias = 256;
        int blocks_bias = (batch_size * out_features + threads_bias - 1) / threads_bias;
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "add_bias_cuda", ([&] {
            add_bias_kernel<scalar_t><<<blocks_bias, threads_bias>>>(
                output.data_ptr<scalar_t>(),
                bias->data_ptr<scalar_t>(),
                batch_size,
                out_features
            );
        }));
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Reshape output to match original input shape
    if (input.dim() > 2) {
        std::vector<int64_t> output_shape;
        for (int i = 0; i < input.dim() - 1; ++i) {
            output_shape.push_back(original_shape[i]);
        }
        output_shape.push_back(out_features);
        output = output.reshape(output_shape);
    } else if (input.dim() == 1) {
        output = output.squeeze(0);
    }
    
    return output;
}

// [END kernel.cu]