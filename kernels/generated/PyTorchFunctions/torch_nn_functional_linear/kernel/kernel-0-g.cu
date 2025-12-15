// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
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
    // Each thread computes one element of the output
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_elements = batch_size * out_features;
    
    if (idx < total_elements) {
        const int64_t batch_idx = idx / out_features;
        const int64_t out_idx = idx % out_features;
        
        scalar_t sum = 0;
        
        // Compute dot product: input[batch_idx, :] . weight[out_idx, :]
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
    TORCH_CHECK(input.dtype() == weight.dtype(), "input and weight must have the same dtype");
    
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias->dtype() == input.dtype(), "bias must have the same dtype as input");
    }
    
    // Handle input shape - can be [*, in_features] where * can be multiple dimensions
    auto input_sizes = input.sizes();
    TORCH_CHECK(input_sizes.size() >= 1, "input must have at least 1 dimension");
    
    const int64_t in_features = input_sizes[input_sizes.size() - 1];
    
    // Flatten all dimensions except the last one
    int64_t batch_size = 1;
    for (int64_t i = 0; i < input_sizes.size() - 1; ++i) {
        batch_size *= input_sizes[i];
    }
    
    // Weight is [out_features, in_features]
    auto weight_sizes = weight.sizes();
    TORCH_CHECK(weight_sizes.size() == 2, "weight must be 2-dimensional");
    const int64_t out_features = weight_sizes[0];
    const int64_t weight_in_features = weight_sizes[1];
    
    TORCH_CHECK(in_features == weight_in_features, 
                "input features must match weight input features");
    
    if (bias.has_value()) {
        auto bias_sizes = bias->sizes();
        TORCH_CHECK(bias_sizes.size() == 1, "bias must be 1-dimensional");
        TORCH_CHECK(bias_sizes[0] == out_features, "bias size must match weight output features");
    }
    
    // 2. Output tensor creation
    // Output shape: [*, out_features]
    std::vector<int64_t> output_sizes;
    for (int64_t i = 0; i < input_sizes.size() - 1; ++i) {
        output_sizes.push_back(input_sizes[i]);
    }
    output_sizes.push_back(out_features);
    
    auto output = torch::empty(output_sizes, input.options());
    
    // 3. Kernel launch parameters
    const int64_t total_elements = batch_size * out_features;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    const bool has_bias = bias.has_value();
    
    // 4. Launch kernel with dtype dispatch
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "linear_kernel", [&] {
        linear_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            has_bias ? bias->data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            in_features,
            out_features,
            has_bias
        );
    });
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch error: ", cudaGetErrorString(err));
    
    // 5. Return tensor
    return output;
}

// [END kernel.cu]