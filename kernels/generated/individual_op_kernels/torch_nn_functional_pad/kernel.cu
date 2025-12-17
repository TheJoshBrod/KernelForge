// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <string>

// ============ DEVICE CODE (CUDA kernels only) ============

template <typename T>
__global__ void constant_pad_kernel(
    const T* input,
    T* output,
    int64_t batch,
    int64_t channels,
    int64_t in_dim1,
    int64_t in_dim2,
    int64_t in_dim3,
    int64_t out_dim1,
    int64_t out_dim2,
    int64_t out_dim3,
    int64_t pad_dim3_left,
    int64_t pad_dim2_left,
    int64_t pad_dim1_left,
    T value
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch * channels * out_dim1 * out_dim2 * out_dim3;
    
    if (idx >= total_elements) return;
    
    // Compute output indices
    int64_t d3_out = idx % out_dim3;
    int64_t d2_out = (idx / out_dim3) % out_dim2;
    int64_t d1_out = (idx / (out_dim3 * out_dim2)) % out_dim1;
    int64_t c = (idx / (out_dim3 * out_dim2 * out_dim1)) % channels;
    int64_t b = idx / (out_dim3 * out_dim2 * out_dim1 * channels);
    
    // Compute corresponding input indices
    int64_t d3_in = d3_out - pad_dim3_left;
    int64_t d2_in = d2_out - pad_dim2_left;
    int64_t d1_in = d1_out - pad_dim1_left;
    
    // Check if we're in the padding region
    if (d3_in < 0 || d3_in >= in_dim3 ||
        d2_in < 0 || d2_in >= in_dim2 ||
        d1_in < 0 || d1_in >= in_dim1) {
        output[idx] = value;
    } else {
        int64_t input_idx = ((b * channels + c) * in_dim1 + d1_in) * in_dim2 * in_dim3 +
                           d2_in * in_dim3 + d3_in;
        output[idx] = input[input_idx];
    }
}

// ============ HOST CODE ============

torch::Tensor launch(
    torch::Tensor input,
    std::vector<int64_t> pad,
    std::string mode,
    c10::optional<double> value
) {
    // 1. Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    input = input.contiguous();
    TORCH_CHECK(input.dim() >= 1, "input must be at least 1D tensor");
    TORCH_CHECK(mode == "constant", "Only 'constant' padding mode is supported");
    
    int64_t ndim = input.dim();
    
    // 2. Get padding value (default to 0.0 if not provided)
    double pad_value = value.has_value() ? value.value() : 0.0;
    
    // 3. Determine dimensions to pad based on pad vector size
    // pad applies to last n//2 dimensions, where n = pad.size()
    int64_t num_pad_dims = pad.size() / 2;
    TORCH_CHECK(num_pad_dims <= ndim, "Padding size must not exceed input dimensions");
    
    // 4. Build output shape
    auto input_shape = input.sizes().vec();
    std::vector<int64_t> output_shape = input_shape;
    
    // Apply padding to the last num_pad_dims dimensions
    // pad format: [left, right, top, bottom, front, back, ...]
    for (int64_t i = 0; i < num_pad_dims; i++) {
        int64_t dim_idx = ndim - 1 - i;  // Start from last dimension
        int64_t pad_left = pad[2 * i];
        int64_t pad_right = pad[2 * i + 1];
        output_shape[dim_idx] = input_shape[dim_idx] + pad_left + pad_right;
        TORCH_CHECK(output_shape[dim_idx] > 0, "Output dimension must be positive");
    }
    
    // 5. Create output tensor
    auto output = torch::zeros(output_shape, input.options());
    
    // 6. Flatten batch dimensions (everything before last 3 dims)
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 3; i++) {
        batch_size *= input_shape[i];
    }
    
    // Get the last 3 dimensions for input
    int64_t in_dim1 = (ndim >= 3) ? input_shape[ndim - 3] : 1;
    int64_t in_dim2 = (ndim >= 2) ? input_shape[ndim - 2] : 1;
    int64_t in_dim3 = input_shape[ndim - 1];
    
    // Get the last 3 dimensions for output
    int64_t out_dim1 = (ndim >= 3) ? output_shape[ndim - 3] : 1;
    int64_t out_dim2 = (ndim >= 2) ? output_shape[ndim - 2] : 1;
    int64_t out_dim3 = output_shape[ndim - 1];
    
    // Get padding for last 3 dimensions (pad vector goes right to left)
    int64_t pad_dim3_left = (pad.size() >= 2) ? pad[0] : 0;  // rightmost dimension
    int64_t pad_dim2_left = (pad.size() >= 4) ? pad[2] : 0;  // second from right
    int64_t pad_dim1_left = (pad.size() >= 6) ? pad[4] : 0;  // third from right
    
    // 7. Calculate grid and block dimensions
    int64_t total_elements = batch_size * out_dim1 * out_dim2 * out_dim3;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    // 8. Launch kernel with dtype dispatch
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "constant_pad_kernel", ([&] {
        constant_pad_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            1,  // channels (folded into batch)
            in_dim1,
            in_dim2,
            in_dim3,
            out_dim1,
            out_dim2,
            out_dim3,
            pad_dim3_left,
            pad_dim2_left,
            pad_dim1_left,
            scalar_t(pad_value)
        );
    }));
    
    // 9. Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    
    return output;
}

// [END kernel.cu]