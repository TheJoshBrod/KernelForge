#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ============ DEVICE CODE (CUDA kernels only) ============

template <typename scalar_t>
__global__ void embedding_kernel(
    const int64_t* __restrict__ indices,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int64_t num_indices,
    int64_t embedding_dim,
    int64_t num_embeddings,
    int64_t padding_idx,
    bool has_padding_idx
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t total_elements = num_indices * embedding_dim;
    
    for (int64_t i = idx; i < total_elements; i += stride) {
        int64_t batch_idx = i / embedding_dim;
        int64_t embed_idx = i % embedding_dim;
        
        int64_t index = indices[batch_idx];
        
        scalar_t value = scalar_t(0);
        
        if (!(has_padding_idx && index == padding_idx)) {
            if (index >= 0 && index < num_embeddings) {
                value = weight[index * embedding_dim + embed_idx];
            }
        }
        
        output[i] = value;
    }
}

// ============ HOST CODE ============

torch::Tensor launch(
    torch::Tensor arg0,
    torch::Tensor arg1,
    c10::optional<int64_t> arg2,
    c10::optional<double> arg3,
    double arg4,
    bool arg5,
    bool arg6
) {
    // Input validation
    TORCH_CHECK(arg0.is_cuda(), "indices tensor must be a CUDA tensor");
    TORCH_CHECK(arg1.is_cuda(), "weight tensor must be a CUDA tensor");
    TORCH_CHECK(arg0.scalar_type() == torch::kLong || arg0.scalar_type() == torch::kInt,
                "indices must be of integer type");
    
    // Ensure contiguous
    auto indices = arg0.contiguous();
    auto weight = arg1.contiguous();
    
    // Extract parameters
    bool has_padding_idx = arg2.has_value();
    int64_t padding_idx = has_padding_idx ? arg2.value() : -1;
    
    // Get dimensions
    auto indices_sizes = indices.sizes();
    auto weight_sizes = weight.sizes();
    
    int64_t num_embeddings = weight_sizes[0];
    int64_t embedding_dim = weight_sizes[1];
    
    // Calculate output shape: same as indices shape + embedding_dim
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < indices.dim(); ++i) {
        output_shape.push_back(indices_sizes[i]);
    }
    output_shape.push_back(embedding_dim);
    
    // Flatten indices for easier indexing
    auto indices_flat = indices.reshape({-1});
    int64_t num_indices = indices_flat.size(0);
    
    // Create output tensor
    auto output = torch::empty(output_shape, weight.options());
    auto output_flat = output.reshape({num_indices, embedding_dim});
    
    // Launch kernel
    int64_t total_elements = num_indices * embedding_dim;
    int threads = 256;
    int blocks = min((total_elements + threads - 1) / threads, (int64_t)2048);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(weight.scalar_type(), "embedding_kernel", [&] {
        embedding_kernel<scalar_t><<<blocks, threads>>>(
            indices_flat.data_ptr<int64_t>(),
            weight.data_ptr<scalar_t>(),
            output_flat.data_ptr<scalar_t>(),
            num_indices,
            embedding_dim,
            num_embeddings,
            padding_idx,
            has_padding_idx
        );
    });
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "CUDA synchronize failed: ", cudaGetErrorString(err));
    
    return output;
}