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
    const int64_t batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    if (batch_idx >= num_indices) return;
    
    const int64_t index = indices[batch_idx];
    const bool is_valid = (!has_padding_idx || index != padding_idx) && 
                          (index >= 0 && index < num_embeddings);
    
    const scalar_t* src = weight + index * embedding_dim;
    scalar_t* dst = output + batch_idx * embedding_dim;
    
    // Vectorized loading for coalesced memory access
    const int vec_size = (sizeof(float4) / sizeof(scalar_t));
    const int64_t vec_dim = embedding_dim / vec_size;
    const int64_t remainder = embedding_dim % vec_size;
    
    if (sizeof(scalar_t) == sizeof(float) && (embedding_dim % 4 == 0)) {
        // Use float4 for float type when aligned
        const float4* src_vec = reinterpret_cast<const float4*>(src);
        float4* dst_vec = reinterpret_cast<float4*>(dst);
        
        for (int64_t i = tid; i < vec_dim; i += blockDim.x) {
            if (is_valid) {
                dst_vec[i] = src_vec[i];
            } else {
                dst_vec[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
    } else {
        // Standard path with coalesced access
        for (int64_t i = tid; i < embedding_dim; i += blockDim.x) {
            if (is_valid) {
                dst[i] = src[i];
            } else {
                dst[i] = scalar_t(0);
            }
        }
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
    
    int threads = (embedding_dim < 256) ? ((embedding_dim + 31) / 32) * 32 : 256;
    threads = min(threads, 1024);
    int blocks = num_indices;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(weight.scalar_type(), "embedding_kernel_optimized", [&] {
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
    });
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "CUDA synchronize failed: ", cudaGetErrorString(err));
    
    return output;
}