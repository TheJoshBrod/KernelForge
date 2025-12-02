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
    const int64_t batch_idx = blockIdx.x;
    const int embed_idx = threadIdx.x;
    
    if (batch_idx >= num_indices) return;
    
    const int64_t index = indices[batch_idx];
    
    const bool is_padding = has_padding_idx && (index == padding_idx);
    const bool is_valid = (index >= 0) && (index < num_embeddings);
    
    for (int64_t e = embed_idx; e < embedding_dim; e += blockDim.x) {
        scalar_t value = scalar_t(0);
        if (!is_padding && is_valid) {
            value = weight[index * embedding_dim + e];
        }
        output[batch_idx * embedding_dim + e] = value;
    }
}

template <typename scalar_t>
__global__ void embedding_kernel_vec4(
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
    
    if (batch_idx >= num_indices) return;
    
    const int64_t index = indices[batch_idx];
    const bool is_padding = has_padding_idx && (index == padding_idx);
    const bool is_valid = (index >= 0) && (index < num_embeddings);
    
    constexpr int VEC_SIZE = 4;
    const int64_t num_vec = embedding_dim / VEC_SIZE;
    
    using vec_t = typename std::conditional<std::is_same<scalar_t, float>::value, float4, 
                  typename std::conditional<std::is_same<scalar_t, double>::value, double4, float4>::type>::type;
    
    const vec_t* weight_vec = reinterpret_cast<const vec_t*>(weight);
    vec_t* output_vec = reinterpret_cast<vec_t*>(output);
    
    for (int64_t v = tid; v < num_vec; v += blockDim.x) {
        vec_t value;
        if (!is_padding && is_valid) {
            value = weight_vec[index * num_vec + v];
        } else {
            if (std::is_same<scalar_t, float>::value) {
                value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            } else {
                value = make_double4(0.0, 0.0, 0.0, 0.0);
            }
        }
        output_vec[batch_idx * num_vec + v] = value;
    }
    
    const int64_t remainder_start = num_vec * VEC_SIZE;
    for (int64_t e = remainder_start + tid; e < embedding_dim; e += blockDim.x) {
        scalar_t value = scalar_t(0);
        if (!is_padding && is_valid) {
            value = weight[index * embedding_dim + e];
        }
        output[batch_idx * embedding_dim + e] = value;
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
    
    int threads = (embedding_dim >= 256) ? 256 : ((embedding_dim >= 128) ? 128 : 64);
    int blocks = num_indices;
    
    bool use_vectorized = (embedding_dim % 4 == 0) && 
                          ((uintptr_t)weight.data_ptr() % 16 == 0) &&
                          ((uintptr_t)output_flat.data_ptr() % 16 == 0);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(weight.scalar_type(), "embedding_kernel", [&] {
        if (use_vectorized && (std::is_same<scalar_t, float>::value || std::is_same<scalar_t, double>::value)) {
            embedding_kernel_vec4<scalar_t><<<blocks, threads>>>(
                indices_flat.data_ptr<int64_t>(),
                weight.data_ptr<scalar_t>(),
                output_flat.data_ptr<scalar_t>(),
                num_indices,
                embedding_dim,
                num_embeddings,
                padding_idx,
                has_padding_idx
            );
        } else {
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
        }
    });
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "CUDA synchronize failed: ", cudaGetErrorString(err));
    
    return output;
}