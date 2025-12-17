// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ============ DEVICE CODE (CUDA kernels only) ============

template <typename scalar_t>
__global__ void embedding_kernel(
    const int64_t* __restrict__ indices,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int64_t num_embeddings,
    const int64_t embedding_dim,
    const int64_t num_indices,
    const int64_t padding_idx,
    const bool has_padding_idx,
    const bool scale_grad_by_freq,
    const bool sparse)
{
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_elements = num_indices * embedding_dim;
    
    if (idx < total_elements) {
        const int64_t seq_idx = idx / embedding_dim;
        const int64_t emb_idx = idx % embedding_dim;
        
        int64_t input_idx = indices[seq_idx];
        
        // Handle negative indices (wrap around)
        if (input_idx < 0) {
            input_idx += num_embeddings;
        }
        
        // Bounds checking
        if (input_idx >= 0 && input_idx < num_embeddings) {
            // Check if this is the padding index
            if (has_padding_idx && input_idx == padding_idx) {
                output[idx] = static_cast<scalar_t>(0);
            } else {
                output[idx] = weight[input_idx * embedding_dim + emb_idx];
            }
        } else {
            // Out of bounds - set to zero
            output[idx] = static_cast<scalar_t>(0);
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
    bool sparse)
{
    // 1. Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2-dimensional");
    
    // Ensure input is contiguous and of type long (int64)
    auto input_contig = input.contiguous();
    if (input_contig.scalar_type() != torch::kLong) {
        input_contig = input_contig.to(torch::kLong);
    }
    
    auto weight_contig = weight.contiguous();
    
    const int64_t num_embeddings = weight_contig.size(0);
    const int64_t embedding_dim = weight_contig.size(1);
    
    // Flatten input to 1D for processing
    auto input_flat = input_contig.view({-1});
    const int64_t num_indices = input_flat.numel();
    
    // 2. Output tensor creation
    auto output_shape = input_contig.sizes().vec();
    output_shape.push_back(embedding_dim);
    auto output = torch::empty(output_shape, weight_contig.options());
    
    // 3. Handle padding_idx
    int64_t padding_idx_value = -1;
    bool has_padding_idx = padding_idx.has_value();
    
    if (has_padding_idx) {
        padding_idx_value = padding_idx.value();
        if (padding_idx_value < 0) {
            padding_idx_value += num_embeddings;
        }
    }
    
    // 4. Kernel launch parameters
    const int threads = 256;
    const int64_t total_elements = num_indices * embedding_dim;
    const int blocks = (total_elements + threads - 1) / threads;
    
    // Note: max_norm is used for gradient clipping during training (forward pass doesn't use it)
    // scale_grad_by_freq and sparse are also for gradients
    
    // 5. Launch kernel with dtype dispatch
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(weight_contig.scalar_type(), "embedding_kernel", [&] {
        embedding_kernel<scalar_t><<<blocks, threads>>>(
            input_flat.data_ptr<int64_t>(),
            weight_contig.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_embeddings,
            embedding_dim,
            num_indices,
            padding_idx_value,
            has_padding_idx,
            scale_grad_by_freq,
            sparse
        );
    });
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA synchronization failed: ", cudaGetErrorString(err));
    
    // 6. Return tensor
    return output;
}

// NO PYBIND11_MODULE HERE!
// [END kernel.cu]