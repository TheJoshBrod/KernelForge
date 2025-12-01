#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ============ DEVICE CODE (CUDA kernels only) ============

template <typename T>
__global__ void constant_pad_3d_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int64_t batch,
    const int64_t channels,
    const int64_t in_depth,
    const int64_t in_height,
    const int64_t in_width,
    const int64_t out_depth,
    const int64_t out_height,
    const int64_t out_width,
    const int64_t pad_left,
    const int64_t pad_top,
    const int64_t pad_front,
    const T pad_value
) {
    const int64_t total_elements = batch * channels * out_depth * out_height * out_width;
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;
    
    // Decompose flat index to 5D coordinates
    int64_t tmp = idx;
    const int64_t w = tmp % out_width;
    tmp /= out_width;
    const int64_t h = tmp % out_height;
    tmp /= out_height;
    const int64_t d = tmp % out_depth;
    tmp /= out_depth;
    const int64_t c = tmp % channels;
    const int64_t b = tmp / channels;
    
    // Calculate input coordinates
    const int64_t in_w = w - pad_left;
    const int64_t in_h = h - pad_top;
    const int64_t in_d = d - pad_front;
    
    // Check if we're in the padding region
    if (in_w < 0 || in_w >= in_width ||
        in_h < 0 || in_h >= in_height ||
        in_d < 0 || in_d >= in_depth) {
        output[idx] = pad_value;
    } else {
        // Calculate input index
        const int64_t in_idx = ((((b * channels + c) * in_depth + in_d) * in_height + in_h) * in_width + in_w);
        output[idx] = input[in_idx];
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
    TORCH_CHECK(input.dim() == 4, "input must be 4D");
    TORCH_CHECK(pad.size() == 6, "pad must have 6 elements (left, right, top, bottom, front, back)");
    TORCH_CHECK(mode == "constant", "only constant mode is supported in this implementation");
    
    // Extract pad value (default to 0.0 if not provided)
    const double pad_value = value.has_value() ? value.value() : 0.0;
    
    // Extract padding values
    const int64_t pad_left = pad[0];
    const int64_t pad_right = pad[1];
    const int64_t pad_top = pad[2];
    const int64_t pad_bottom = pad[3];
    const int64_t pad_front = pad[4];
    const int64_t pad_back = pad[5];
    
    // Get input dimensions [B, C, H, W]
    const int64_t batch = input.size(0);
    const int64_t channels = input.size(1);
    const int64_t in_height = input.size(2);
    const int64_t in_width = input.size(3);
    
    // For 4D input with 6-element padding, treat as [B, C, D=1, H, W]
    const int64_t in_depth = 1;
    const int64_t in_h = in_height;
    const int64_t in_w = in_width;
    
    // Calculate output dimensions
    const int64_t out_depth = in_depth + pad_front + pad_back;
    const int64_t out_height = in_h + pad_top + pad_bottom;
    const int64_t out_width = in_w + pad_left + pad_right;
    
    // Create output tensor (5D internally)
    auto output = torch::empty({batch, channels, out_depth, out_height, out_width}, input.options());
    
    // Calculate total elements
    const int64_t total_elements = batch * channels * out_depth * out_height * out_width;
    
    // Kernel launch parameters - optimized for GTX 1660 Ti (Turing architecture)
    const int threads = 256;  // Good occupancy for Turing
    const int blocks = (total_elements + threads - 1) / threads;
    
    // Dispatch based on dtype
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "constant_pad_3d_kernel", ([&] {
        constant_pad_3d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch,
            channels,
            in_depth,
            in_h,
            in_w,
            out_depth,
            out_height,
            out_width,
            pad_left,
            pad_top,
            pad_front,
            scalar_t(pad_value)  // Use the actual pad_value parameter
        );
    }));
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    // Squeeze out the depth dimension if it's 1 (to return 4D for 4D input)
    if (out_depth == 1) {
        output = output.squeeze(2);  // Remove dimension 2 (depth)
    }
    
    return output;
}

// NO PYBIND11_MODULE HERE!
// [END kernel.cu]