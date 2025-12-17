// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <string>

// ============ DEVICE CODE (CUDA kernels only) ============

template <typename T>
__global__ void constant_pad_3d_kernel(
    const T* input,
    T* output,
    int64_t batch,
    int64_t channels,
    int64_t in_depth,
    int64_t in_height,
    int64_t in_width,
    int64_t out_depth,
    int64_t out_height,
    int64_t out_width,
    int64_t pad_left,
    int64_t pad_top,
    int64_t pad_front,
    T value
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch * channels * out_depth * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    // Compute output indices
    int64_t w_out = idx % out_width;
    int64_t h_out = (idx / out_width) % out_height;
    int64_t d_out = (idx / (out_width * out_height)) % out_depth;
    int64_t c = (idx / (out_width * out_height * out_depth)) % channels;
    int64_t b = idx / (out_width * out_height * out_depth * channels);
    
    // Compute corresponding input indices
    int64_t w_in = w_out - pad_left;
    int64_t h_in = h_out - pad_top;
    int64_t d_in = d_out - pad_front;
    
    // Check if we're in the padding region
    if (w_in < 0 || w_in >= in_width ||
        h_in < 0 || h_in >= in_height ||
        d_in < 0 || d_in >= in_depth) {
        output[idx] = value;
    } else {
        int64_t input_idx = ((b * channels + c) * in_depth + d_in) * in_height * in_width +
                           h_in * in_width + w_in;
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
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must be 4D tensor (N, C, D, H, W)");
    TORCH_CHECK(pad.size() == 6, "pad must have 6 elements");
    TORCH_CHECK(mode == "constant", "Only 'constant' padding mode is supported");
    
    // 2. Extract input dimensions
    int64_t batch = input.size(0);
    int64_t channels = input.size(1);
    int64_t in_depth = input.size(2);
    int64_t in_height = input.size(3);
    int64_t in_width = input.size(4);
    
    // 3. Parse padding (left, right, top, bottom, front, back)
    int64_t pad_left = pad[0];
    int64_t pad_right = pad[1];
    int64_t pad_top = pad[2];
    int64_t pad_bottom = pad[3];
    int64_t pad_front = pad[4];
    int64_t pad_back = pad[5];
    
    // 4. Calculate output dimensions
    int64_t out_width = in_width + pad_left + pad_right;
    int64_t out_height = in_height + pad_top + pad_bottom;
    int64_t out_depth = in_depth + pad_front + pad_back;
    
    TORCH_CHECK(out_width > 0 && out_height > 0 && out_depth > 0,
                "output dimensions must be positive");
    
    // 5. Get padding value (default to 0.0 if not provided)
    double pad_value = value.has_value() ? value.value() : 0.0;
    
    // 6. Create output tensor
    auto output = torch::zeros({batch, channels, out_depth, out_height, out_width}, input.options());
    
    // 7. Calculate grid and block dimensions
    int64_t total_elements = batch * channels * out_depth * out_height * out_width;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    // 8. Launch kernel with dtype dispatch
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "constant_pad_3d_kernel", ([&] {
        constant_pad_3d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch,
            channels,
            in_depth,
            in_height,
            in_width,
            out_depth,
            out_height,
            out_width,
            pad_left,
            pad_top,
            pad_front,
            scalar_t(pad_value)
        );
    }));
    
    // 9. Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    
    return output;
}

// [END kernel.cu]