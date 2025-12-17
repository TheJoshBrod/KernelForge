// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ============ DEVICE CODE (CUDA kernels only) ============

template <typename T>
__global__ void adaptive_avg_pool1d_kernel(
    const T* input,
    T* output,
    int64_t batch_size,
    int64_t channels,
    int64_t input_size,
    int64_t output_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_threads = batch_size * channels * output_size;
    
    if (idx >= total_threads) return;
    
    // Decode index to (b, c, o)
    int64_t o = idx % output_size;
    int64_t c = (idx / output_size) % channels;
    int64_t b = idx / (output_size * channels);
    
    // Calculate input range for this output position
    int64_t start_idx = (o * input_size) / output_size;
    int64_t end_idx = ((o + 1) * input_size + output_size - 1) / output_size;
    
    // Accumulate sum
    T sum = 0;
    int64_t count = 0;
    
    for (int64_t i = start_idx; i < end_idx; ++i) {
        int64_t input_idx = b * channels * input_size + c * input_size + i;
        sum += input[input_idx];
        count++;
    }
    
    // Compute average
    int64_t output_idx = b * channels * output_size + c * output_size + o;
    output[output_idx] = (count > 0) ? (sum / static_cast<T>(count)) : static_cast<T>(0);
}

// ============ HOST CODE ============

torch::Tensor launch(torch::Tensor input, int64_t output_size) {
    // 1. Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "input must be 3-dimensional (N, C, L)");
    
    // Make input contiguous
    input = input.contiguous();
    
    // 2. Extract dimensions
    int64_t batch_size = input.size(0);
    int64_t channels = input.size(1);
    int64_t input_size = input.size(2);
    
    // Validate output_size
    TORCH_CHECK(output_size > 0, "output_size must be positive");
    
    // 3. Output tensor creation
    auto output = torch::empty({batch_size, channels, output_size}, 
                               torch::TensorOptions()
                                   .dtype(input.dtype())
                                   .device(input.device()));
    
    // 4. Kernel launch parameters
    int64_t total_threads = batch_size * channels * output_size;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;
    
    // 5. Launch kernel with dtype dispatch
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "adaptive_avg_pool1d_kernel", [&] {
        adaptive_avg_pool1d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_size,
            output_size
        );
    });
    
    // 6. Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    // 7. Return tensor
    return output;
}

// [END kernel.cu]