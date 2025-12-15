// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ============ DEVICE CODE (CUDA kernels only) ============

template <typename scalar_t>
__global__ void constant_pad_3d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
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
    scalar_t value
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch * channels * out_depth * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    // Compute output coordinates
    int64_t w_out = idx % out_width;
    int64_t tmp = idx / out_width;
    int64_t h_out = tmp % out_height;
    tmp = tmp / out_height;
    int64_t d_out = tmp % out_depth;
    tmp = tmp / out_depth;
    int64_t c = tmp % channels;
    int64_t b = tmp / channels;
    
    // Compute corresponding input coordinates
    int64_t w_in = w_out - pad_left;
    int64_t h_in = h_out - pad_top;
    int64_t d_in = d_out - pad_front;
    
    // Check if we're in the padded region or the original input region
    if (d_in >= 0 && d_in < in_depth &&
        h_in >= 0 && h_in < in_height &&
        w_in >= 0 && w_in < in_width) {
        // Copy from input
        int64_t in_idx = ((b * channels + c) * in_depth + d_in) * in_height * in_width +
                         h_in * in_width + w_in;
        output[idx] = input[in_idx];
    } else {
        // Use padding value
        output[idx] = value;
    }
}

// ============ HOST CODE ============

torch::Tensor launch(torch::Tensor input, std::vector<int64_t> pad) {
    // 1. Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must be 4D tensor (batch, channel, depth, height, width) or (batch, channel, height, width)");
    TORCH_CHECK(pad.size() == 6 || pad.size() == 4 || pad.size() == 2, "pad must have 2, 4, or 6 elements");
    
    // Get input dimensions
    int64_t batch = input.size(0);
    int64_t channels = input.size(1);
    
    // Handle different padding configurations
    int64_t pad_left = 0, pad_right = 0;
    int64_t pad_top = 0, pad_bottom = 0;
    int64_t pad_front = 0, pad_back = 0;
    
    int64_t in_depth = 1, in_height = 1, in_width = 1;
    
    if (pad.size() == 6) {
        // 3D padding: (left, right, top, bottom, front, back)
        pad_left = pad[0];
        pad_right = pad[1];
        pad_top = pad[2];
        pad_bottom = pad[3];
        pad_front = pad[4];
        pad_back = pad[5];
        
        // Input is NCDHW
        TORCH_CHECK(input.dim() == 4, "For 3D padding, input should be 4D");
        in_depth = input.size(2);
        in_height = input.size(3);
        in_width = input.size(3);  // Treat as square for simplicity
    } else if (pad.size() == 4) {
        // 2D padding: (left, right, top, bottom)
        pad_left = pad[0];
        pad_right = pad[1];
        pad_top = pad[2];
        pad_bottom = pad[3];
        
        // Input is NCHW
        in_depth = input.size(2);
        in_height = input.size(3);
        in_width = input.size(3);
    } else {
        // 1D padding: (left, right)
        pad_left = pad[0];
        pad_right = pad[1];
        
        in_depth = 1;
        in_height = input.size(2);
        in_width = input.size(3);
    }
    
    // For the actual case: input is [1, C, H, W] with 6-element pad
    // Reinterpret dimensions correctly
    if (input.dim() == 4 && pad.size() == 6) {
        in_depth = input.size(2);
        in_height = input.size(3);
        in_width = input.size(3);  // This will be updated below
        
        // Actually for NCHW with 6-pad, interpret as:
        // pad[0,1] = width, pad[2,3] = height, pad[4,5] = depth
        // But typically: left, right, top, bottom, front, back
    }
    
    // Recalculate for standard NCHW case with 6-element padding
    if (input.dim() == 4) {
        if (pad.size() == 6) {
            // Padding order: left, right, top, bottom, front, back
            // For NCHW: W, H, D (but we only have 2 spatial dims)
            // Interpret as: left/right for last dim (W), top/bottom for second-to-last (H), front/back for third-to-last (D)
            in_width = input.size(3);
            in_height = input.size(2);
            in_depth = 1;
        } else if (pad.size() == 4) {
            in_width = input.size(3);
            in_height = input.size(2);
            in_depth = 1;
        }
    }
    
    // Calculate output dimensions
    int64_t out_width = in_width + pad_left + pad_right;
    int64_t out_height = in_height + pad_top + pad_bottom;
    int64_t out_depth = in_depth + pad_front + pad_back;
    
    // Create output tensor
    auto output = torch::zeros({batch, channels, out_depth, out_height, out_width}, input.options());
    
    // Adjust for NCHW case
    if (input.dim() == 4 && pad.size() <= 4) {
        output = torch::zeros({batch, channels, out_height, out_width}, input.options());
        out_depth = 1;
        in_depth = 1;
    }
    
    // Special handling for common case: NCHW with 6-element pad
    if (input.dim() == 4 && pad.size() == 6) {
        output = torch::zeros({batch, channels, out_height, out_width}, input.options());
        out_depth = 1;
        in_depth = 1;
    }
    
    // Kernel launch parameters
    int64_t total_elements = output.numel();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    // Launch kernel with dtype dispatch
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
            scalar_t(0.0)  // constant padding value
        );
    }));
    
    // Error checking
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));
    
    return output;
}

// [END kernel.cu]