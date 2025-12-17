// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ============ DEVICE CODE (CUDA kernels only) ============

template <typename scalar_t>
__global__ void adaptive_avg_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width
) {
    // Each thread handles one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * output_height * output_width;
    
    if (idx >= total_elements) return;
    
    // Decode indices
    int ow = idx % output_width;
    int oh = (idx / output_width) % output_height;
    int c = (idx / (output_width * output_height)) % channels;
    int b = idx / (output_width * output_height * channels);
    
    // Compute input pooling region
    int ih_start = (oh * input_height) / output_height;
    int ih_end = ((oh + 1) * input_height + output_height - 1) / output_height;
    
    int iw_start = (ow * input_width) / output_width;
    int iw_end = ((ow + 1) * input_width + output_width - 1) / output_width;
    
    // Clamp to valid range
    ih_end = min(ih_end, input_height);
    iw_end = min(iw_end, input_width);
    
    // Compute average over the region
    scalar_t sum = 0;
    int count = 0;
    
    for (int ih = ih_start; ih < ih_end; ++ih) {
        for (int iw = iw_start; iw < iw_end; ++iw) {
            int input_idx = b * (channels * input_height * input_width) +
                          c * (input_height * input_width) +
                          ih * input_width +
                          iw;
            sum += input[input_idx];
            count++;
        }
    }
    
    output[idx] = (count > 0) ? (sum / static_cast<scalar_t>(count)) : static_cast<scalar_t>(0);
}

// ============ HOST CODE ============

torch::Tensor launch(torch::Tensor input, std::vector<int64_t> output_size) {
    // 1. Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (NCHW)");
    TORCH_CHECK(output_size.size() == 2, "output_size must have 2 elements");
    
    // Make contiguous
    input = input.contiguous();
    
    // Extract dimensions
    int64_t batch_size = input.size(0);
    int64_t channels = input.size(1);
    int64_t input_height = input.size(2);
    int64_t input_width = input.size(3);
    int64_t output_height = output_size[0];
    int64_t output_width = output_size[1];
    
    // 2. Output tensor creation
    auto output = torch::empty({batch_size, channels, output_height, output_width}, 
                               input.options());
    
    // 3. Kernel launch parameters
    int64_t total_elements = batch_size * channels * output_height * output_width;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    // 4. Launch kernel with dtype dispatch
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "adaptive_avg_pool2d_kernel", ([&] {
        adaptive_avg_pool2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width
        );
    }));
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch error: ", cudaGetErrorString(err));
    
    // 5. Return tensor
    return output;
}

// [END kernel.cu]