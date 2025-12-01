#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ============ DEVICE CODE (CUDA kernels only) ============

template <typename T>
__global__ void embedding_kernel(
    const int64_t* indices,
    const T* weight,
    T* output,
    int64_t num_embeddings,
    int64_t embedding_dim,
    int64_t num_indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = num_indices * embedding_dim;
    
    if (idx < total_elements) {
        int64_t embedding_idx = idx / embedding_dim;
        int64_t feature_idx = idx % embedding_dim;
        
        int64_t weight_idx = indices[embedding_idx];
        
        // Handle negative indices
        if (weight_idx < 0) {
            weight_idx += num_embeddings;
        }
        
        // Bounds check
        if (weight_idx >= 0 && weight_idx < num_embeddings) {
            // Check if this is the padding index (only if padding_idx >= 0)
            if (padding_idx >= 0 && weight_idx == padding_idx) {
                output[idx] = static_cast<T>(0);
            } else {
                output[idx] = weight[weight_idx * embedding_dim + feature_idx];
            }
        } else {
            // Out of bounds - set to 0
            output[idx] = static_cast<T>(0);
        }
    }
}

// ============ HOST CODE ============

torch::Tensor launch(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<int64_t> padding_idx,
    c10::optional<double> max_norm,
    double norm_type,
    bool scale_grad_by_freq,
    bool sparse
) {
    // 1. Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kInt64 || input.dtype() == torch::kInt32 || input.dtype() == torch::kLong,
                "input must be an integer tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor");
    
    // Ensure contiguous
    input = input.contiguous();
    weight = weight.contiguous();
    
    // 2. Extract padding_idx value (use -1 as sentinel for "no padding")
    int64_t padding_idx_val = padding_idx.has_value() ? padding_idx.value() : -1;
    
    // 3. Get dimensions
    auto input_shape = input.sizes().vec();
    int64_t num_embeddings = weight.size(0);
    int64_t embedding_dim = weight.size(1);
    
    // Flatten input to 1D for easier indexing
    auto input_flat = input.reshape({-1});
    int64_t num_indices = input_flat.numel();
    
    // 4. Create output tensor
    auto output_shape = input_shape;
    output_shape.push_back(embedding_dim);
    auto output = torch::empty(output_shape, weight.options());
    auto output_flat = output.reshape({-1});
    
    // 5. Kernel launch parameters
    int64_t total_elements = num_indices * embedding_dim;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    // 6. Launch kernel with dtype dispatch
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(weight.scalar_type(), "embedding_kernel", ([&] {
        embedding_kernel<scalar_t><<<blocks, threads>>>(
            input_flat.data_ptr<int64_t>(),
            weight.data_ptr<scalar_t>(),
            output_flat.data_ptr<scalar_t>(),
            num_embeddings,
            embedding_dim,
            num_indices,
            padding_idx_val,
            scale_grad_by_freq,
            sparse
        );
    }));
    
    // 7. Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));
    
    // 8. Return tensor with proper shape
    return output;
}

// [END kernel.cu]