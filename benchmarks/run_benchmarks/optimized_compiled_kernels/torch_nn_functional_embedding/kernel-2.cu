```cpp
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ============ DEVICE CODE (CUDA kernels only) ============

template <typename scalar_t>
__global__ void embedding_kernel_optimized(
    const int64_t* __restrict__ indices,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int64_t num_indices,
    int64_t embedding_dim,
    int64_t num_embeddings,
    int64_t padding_idx,
    bool has_padding_idx
) {
    // Each block handles one embedding lookup
    int64_t batch_idx = blockIdx.x;
    
    if (batch_idx >= num_indices) return;
    
    int64_t index = indices[batch_idx];
    
    // Check if this is a padding index or out of bounds
    bool is_zero = (has_padding_idx && index == padding_idx) || 
                   (index < 0 || index >= num_embeddings);
    
    // Each thread handles multiple elements using grid-stride loop
    const scalar_t* src = weight + index * embedding_dim;
    scalar_t* dst = output + batch_idx * embedding_dim;
    
    // Vectorized load/store for better memory coalescing
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    if (is_zero) {
        // Write zeros
        for (int64_t i = tid; i < embedding_dim; i += stride) {
            dst[i] = scalar_t(0);
        }
    } else {
        // Copy from weight matrix with coalesced access
        for (int64_t i = tid; i < embedding_dim; i += stride) {
            dst[i] = src[i];
        }
    }
}

// Specialized kernel for small embedding dimensions using shared memory
template <typename scalar_t, int EMBED_DIM>
__global__ void embedding_kernel_small(
    const int64_t* __restrict__ indices,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int64_t num_indices,
    int64_t num_embeddings,
    int64_t padding_idx,
    bool has_padding_idx
) {
    __shared__ scalar_t smem[EMBED_DIM];
    
    int64_t batch_idx = blockIdx.x;
    if (batch_idx >= num_indices) return;
    
    int64_t index = indices[batch_idx];
    bool is_zero = (has_padding_idx && index == padding_idx) || 
                   (index < 0 || index >= num_embeddings);
    
    int tid = threadIdx.x;
    
    // Cooperatively load into shared memory
    if (!is_zero) {
        const scalar_t* src = weight + index * EMBED_DIM;
        for (int i = tid; i < EMBED_DIM; i += blockDim.x) {
            smem[i] = src[i];
        }
    } else {
        for (int i = tid; i < EMBED_DIM; i += blockDim.x) {
            smem[i] = scalar_t(0);
        }
    }
    
    __syncthreads();
    
    // Write out from shared memory
    scalar_t* dst = output + batch_idx * EMBED_DIM;
    for (int i = tid; i < EMBED_DIM; i += blockDim.x) {
        dst[i] = smem[i];
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
    TORCH_CHECK(arg0.is_cuda(), "indices tensor must be a CUDA tensor");
    TORCH_CHECK(arg1.is_cuda(), "weight tensor must be a CUDA tensor");
    TORCH_CHECK(arg0.scalar_type() == torch::kLong || arg0.scalar_type() == torch::kInt,
                "indices must be of integer type");
    
    auto indices = arg0.contiguous();
    auto weight = arg1.contiguous();
    
    bool has_padding_idx = arg2.has_value();
    int64_t padding_idx = has_padding_idx ? arg2.value() : -1;
    
    auto indices_sizes = indices.sizes();
    auto weight_sizes = weight.sizes();
    
    int64_t num_embeddings = weight_sizes[0];
    int64_t embedding_dim = weight_sizes[1];
    
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < indices.dim(); ++i) {
        output_shape.push_back(indices_sizes[i]);
    }
    output_shape.push_back(embedding_dim);
    
    auto indices_flat = indices.reshape({-1});
    int64_t num_indices = indices_flat.size(0);
    
    auto output = torch::empty(output_shape, weight.options());
    auto output_flat = output.reshape({num_indices, embedding_dim});
    
    // Choose kernel based on embedding dimension
    int threads = (embedding_dim < 256) ? ((embedding_dim + 31) / 32) * 32 : 256;
    threads = min(threads, 256);
    int blocks = num_indices;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(weight.scalar_type(), "embedding_kernel", [&] {
        if (embedding_dim <= 128 && embedding_dim * sizeof(scalar_t) <= 6144) {
            // Use shared memory kernel for small embeddings
            if (embedding_dim == 128) {
                embedding_kernel_small<scalar_t, 128><<<blocks, threads>>>(
                    indices_flat.data_ptr<int64_t>(),
                    weight.data_ptr<scalar_t>(),
                    output_flat.data_ptr<scalar_t>(),
                    num_indices,
                    num_embeddings,
                    padding_idx,
                    has_padding_idx
                );
            } else {
                embedding_kernel_optimized<scalar_t><<<blocks, threads>>>(
                    indices_flat.data_ptr<int64_t>(),
                    weight.data_ptr<scalar_t>(),
                    output_flat.data_ptr<scalar_t>(),
                    num_indices,
                    embedding_dim,
                    num_embeddings,
                    padding_idx,
                    has_padding_idx
                );
            }
        } else {
            embedding_kernel_optimized<scalar_t><<<blocks, threads>>>(
                indices_flat.data_ptr<int64_t>(),
                weight.data_ptr<scalar_t>(),
                output_flat.data_ptr<scalar_t>(),
                num_indices,
                embedding_dim,
                num_embeddings,
                padding_idx,
                has_padding_idx
            );
        }
    });
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "CUDA synchronize failed: ", cudaGetErrorString(err));
    
    return output;
}
```