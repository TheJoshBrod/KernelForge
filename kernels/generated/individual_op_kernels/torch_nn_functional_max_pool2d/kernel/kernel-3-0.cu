// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ============ DEVICE CODE (CUDA kernels only) ============

template <typename T>
__global__ void max_pool2d_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const bool return_indices
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * channels * output_height * output_width;
    
    if (idx >= total_elements) return;
    
    // Decompose linear index
    const int ow = idx % output_width;
    const int oh = (idx / output_width) % output_height;
    const int c = (idx / (output_width * output_height)) % channels;
    const int b = idx / (output_width * output_height * channels);
    
    // Calculate input window bounds
    const int h_start = oh * stride_h - pad_h;
    const int w_start = ow * stride_w - pad_w;
    
    T max_val = -__FLT_MAX__;
    int64_t max_idx = -1;
    
    // Iterate over pooling window
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            const int ih = h_start + kh * dilation_h;
            const int iw = w_start + kw * dilation_w;
            
            // Check bounds
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                const int input_idx = ((b * channels + c) * input_height + ih) * input_width + iw;
                const T val = input[input_idx];
                
                if (val > max_val) {
                    max_val = val;
                    max_idx = input_idx;
                }
            }
        }
    }
    
    output[idx] = max_val;
    if (return_indices && indices != nullptr) {
        indices[idx] = max_idx;
    }
}

// ============ HOST CODE ============

torch::Tensor launch(
    torch::Tensor arg0,
    int64_t arg1,
    int64_t arg2,
    int64_t arg3,
    int64_t arg4,
    bool arg5,
    bool arg6
) {
    // Input validation
    TORCH_CHECK(arg0.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(arg0.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(arg0.dim() == 4, "input must be 4D (NCHW)");
    
    auto input = arg0.contiguous();
    
    // Extract dimensions
    const int64_t batch_size = input.size(0);
    const int64_t channels = input.size(1);
    const int64_t input_height = input.size(2);
    const int64_t input_width = input.size(3);
    
    // Parse pooling parameters
    const int64_t kernel_h = arg1;
    const int64_t kernel_w = arg1;
    const int64_t stride_h = arg2;
    const int64_t stride_w = arg2;
    const int64_t pad_h = arg3;
    const int64_t pad_w = arg3;
    const int64_t dilation_h = arg4;
    const int64_t dilation_w = arg4;
    const bool ceil_mode = arg5;
    const bool return_indices = arg6;
    
    // Calculate output dimensions
    int64_t output_height, output_width;
    
    if (ceil_mode) {
        output_height = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1 + stride_h) / stride_h;
        output_width = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1 + stride_w) / stride_w;
    } else {
        output_height = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
        output_width = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    }
    
    // Create output tensors
    auto output = torch::empty({batch_size, channels, output_height, output_width}, 
                               input.options());
    
    torch::Tensor indices;
    int64_t* indices_ptr = nullptr;
    
    if (return_indices) {
        indices = torch::empty({batch_size, channels, output_height, output_width}, 
                               input.options().dtype(torch::kLong));
        indices_ptr = indices.data_ptr<int64_t>();
    }
    
    // Kernel launch parameters
    const int64_t total_elements = batch_size * channels * output_height * output_width;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    // Launch kernel with dtype dispatch
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_pool2d_kernel", [&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            indices_ptr,
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            return_indices
        );
    });
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    // Return only output tensor (not indices even if computed)
    return output;
}

// [END kernel.cu]