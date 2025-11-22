#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ============ DEVICE CODE (CUDA kernels only) ============

template <typename scalar_t>
__global__ void constant_pad_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t batch,
    int64_t channels,
    int64_t in_height,
    int64_t in_width,
    int64_t out_height,
    int64_t out_width,
    int64_t pad_left,
    int64_t pad_top,
    scalar_t value
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch * channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    // Compute output coordinates
    int64_t w_out = idx % out_width;
    int64_t tmp = idx / out_width;
    int64_t h_out = tmp % out_height;
    tmp = tmp / out_height;
    int64_t c = tmp % channels;
    int64_t b = tmp / channels;
    
    // Compute corresponding input coordinates
    int64_t w_in = w_out - pad_left;
    int64_t h_in = h_out - pad_top;
    
    // Check if we're in the padded region or the original input region
    if (h_in >= 0 && h_in < in_height &&
        w_in >= 0 && w_in < in_width) {
        // Copy from input
        int64_t in_idx = ((b * channels + c) * in_height + h_in) * in_width + w_in;
        output[idx] = input[in_idx];
    } else {
        // Use padding value
        output[idx] = value;
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
    TORCH_CHECK(mode == "constant", "only constant mode is supported in this implementation");
    TORCH_CHECK(pad.size() == 6 || pad.size() == 4 || pad.size() == 2, 
                "pad must have 2, 4, or 6 elements");
    
    // Get the padding value (default to 0.0 if not provided)
    double pad_value = value.has_value() ? value.value() : 0.0;
    
    // Get input dimensions
    int64_t batch = input.size(0);
    int64_t channels = input.size(1);
    int64_t in_height = input.size(2);
    int64_t in_width = input.size(3);
    
    // Parse padding values
    // PyTorch pad format: (left, right, top, bottom, front, back)
    // For 2D tensors (NCHW), we use: (left, right, top, bottom)
    int64_t pad_left = 0, pad_right = 0;
    int64_t pad_top = 0, pad_bottom = 0;
    int64_t pad_front = 0, pad_back = 0;
    
    if (pad.size() == 2) {
        // 1D padding: (left, right) - applies to last dimension (width)
        pad_left = pad[0];
        pad_right = pad[1];
    } else if (pad.size() == 4) {
        // 2D padding: (left, right, top, bottom) - applies to last 2 dimensions
        pad_left = pad[0];
        pad_right = pad[1];
        pad_top = pad[2];
        pad_bottom = pad[3];
    } else if (pad.size() == 6) {
        // 3D padding: (left, right, top, bottom, front, back) - applies to last 3 dimensions
        // For 4D input (NCHW), front/back would apply to channels, but that's unusual
        // Standard interpretation: left/right for W, top/bottom for H, front/back for C
        pad_left = pad[0];
        pad_right = pad[1];
        pad_top = pad[2];
        pad_bottom = pad[3];
        pad_front = pad[4];
        pad_back = pad[5];
        
        // For typical NCHW case with 6-element pad, we'll apply:
        // - left/right to width (last dim)
        // - top/bottom to height (second-to-last dim)
        // - front/back to channels (third-to-last dim)
        // But this is unusual. We'll handle the common 2D case.
    }
    
    // Calculate output dimensions
    int64_t out_width = in_width + pad_left + pad_right;
    int64_t out_height = in_height + pad_top + pad_bottom;
    
    // For 6-element padding on 4D tensor, we might need to pad channels too
    int64_t out_channels = channels;
    int64_t out_batch = batch;
    
    if (pad.size() == 6) {
        out_channels = channels + pad_front + pad_back;
    }
    
    // Create output tensor with appropriate shape
    torch::Tensor output;
    if (pad.size() == 6 && pad_front != 0) {
        output = torch::zeros({out_batch, out_channels, out_height, out_width}, input.options());
    } else {
        output = torch::zeros({batch, channels, out_height, out_width}, input.options());
    }
    
    // For simplicity, if we have channel padding (6-element with non-zero front/back),
    // we'll handle it differently. For now, focus on the common case.
    if (pad.size() == 6 && (pad_front != 0 || pad_back != 0)) {
        // Handle channel padding - this is more complex
        // For now, we'll implement a simpler version that handles spatial padding only
        TORCH_CHECK(pad_front == 0 && pad_back == 0, 
                    "Channel padding (front/back in 6-element pad) not fully implemented");
    }
    
    // Kernel launch parameters
    int64_t total_elements = batch * channels * out_height * out_width;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    // Launch kernel with dtype dispatch
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "constant_pad_kernel", ([&] {
        constant_pad_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch,
            channels,
            in_height,
            in_width,
            out_height,
            out_width,
            pad_left,
            pad_top,
            scalar_t(pad_value)
        );
    }));
    
    // Error checking
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));
    
    return output;
}

// [END kernel.cu]