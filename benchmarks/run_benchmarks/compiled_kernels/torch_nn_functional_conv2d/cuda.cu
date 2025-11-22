#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ============ DEVICE CODE (CUDA kernels only) ============

template <typename T>
__global__ void conv2d_kernel(
    T* output,
    const T* input,
    const T* weight,
    const T* bias,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    bool has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    // Decode output position
    int w_out = idx % out_width;
    int h_out = (idx / out_width) % out_height;
    int c_out = (idx / (out_width * out_height)) % out_channels;
    int b = idx / (out_width * out_height * out_channels);
    
    // Determine group
    int channels_per_group = out_channels / groups;
    int group_id = c_out / channels_per_group;
    int in_channels_per_group = in_channels / groups;
    
    T sum = 0;
    
    // Convolution computation
    for (int c_in = 0; c_in < in_channels_per_group; c_in++) {
        int actual_c_in = group_id * in_channels_per_group + c_in;
        
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int h_in = h_out * stride_h - padding_h + kh * dilation_h;
                int w_in = w_out * stride_w - padding_w + kw * dilation_w;
                
                if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                    int input_idx = ((b * in_channels + actual_c_in) * in_height + h_in) * in_width + w_in;
                    int weight_idx = ((c_out * in_channels_per_group + c_in) * kernel_h + kh) * kernel_w + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias if present
    if (has_bias && bias != nullptr) {
        sum += bias[c_out];
    }
    
    output[idx] = sum;
}

// ============ HOST CODE ============

torch::Tensor launch(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups
) {
    // 1. Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (out_channels, in_channels/groups, kH, kW)");
    
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
    }
    
    // 2. Extract dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    
    int stride_h = stride[0];
    int stride_w = stride[1];
    int padding_h = padding[0];
    int padding_w = padding[1];
    int dilation_h = dilation[0];
    int dilation_w = dilation[1];
    
    // 3. Calculate output dimensions
    int out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    // 4. Create output tensor
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    // 5. Launch kernel
    int total_elements = batch_size * out_channels * out_height * out_width;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    bool has_bias = bias.has_value();
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "conv2d_kernel", ([&] {
        conv2d_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            has_bias ? bias->data_ptr<scalar_t>() : nullptr,
            batch_size,
            in_channels,
            in_height,
            in_width,
            out_channels,
            out_height,
            out_width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups,
            has_bias
        );
    }));
    
    // 6. Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    
    cudaDeviceSynchronize();
    
    // 7. Return tensor
    return output;
}

// [END kernel.cu]