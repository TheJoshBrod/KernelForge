// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ============ DEVICE CODE (CUDA kernels only) ============

template <typename scalar_t>
__global__ void conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int out_height,
    const int out_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const bool has_bias
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    // Decode output position
    const int ow = idx % out_width;
    const int oh = (idx / out_width) % out_height;
    const int oc = (idx / (out_width * out_height)) % out_channels;
    const int n = idx / (out_width * out_height * out_channels);
    
    // Compute group parameters
    const int in_channels_per_group = in_channels / groups;
    const int out_channels_per_group = out_channels / groups;
    const int group_id = oc / out_channels_per_group;
    const int oc_in_group = oc % out_channels_per_group;
    const int ic_start = group_id * in_channels_per_group;
    
    scalar_t sum = 0;
    
    // Convolution computation
    for (int ic = 0; ic < in_channels_per_group; ++ic) {
        const int ic_global = ic_start + ic;
        
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int ih = oh * stride_h - padding_h + kh * dilation_h;
                const int iw = ow * stride_w - padding_w + kw * dilation_w;
                
                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    const int input_idx = ((n * in_channels + ic_global) * in_height + ih) * in_width + iw;
                    const int weight_idx = ((oc * in_channels_per_group + ic) * kernel_h + kh) * kernel_w + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias if present
    if (has_bias) {
        sum += bias[oc];
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
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (out_channels, in_channels/groups, kH, kW)");
    
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->dim() == 1, "bias must be 1D");
    }
    
    // Make inputs contiguous
    input = input.contiguous();
    weight = weight.contiguous();
    
    // Extract dimensions
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    
    const int out_channels = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    
    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int padding_h = padding[0];
    const int padding_w = padding[1];
    const int dilation_h = dilation[0];
    const int dilation_w = dilation[1];
    
    // Compute output dimensions
    const int out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int out_width = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    // Create output tensor
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, input.options());
    
    // Kernel launch parameters
    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    const bool has_bias = bias.has_value();
    const void* bias_ptr = has_bias ? bias->data_ptr() : nullptr;
    
    // Launch kernel with dtype dispatch
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "conv2d_kernel", ([&] {
        conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            has_bias ? bias->data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
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
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

// [END kernel.cu]