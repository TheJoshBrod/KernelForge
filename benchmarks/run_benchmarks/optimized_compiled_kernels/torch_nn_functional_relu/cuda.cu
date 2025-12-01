#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ============ DEVICE CODE (CUDA kernels only) ============

template <typename T>
__global__ void relu_kernel(T* output, const T* input, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        T val = input[idx];
        output[idx] = val > T(0) ? val : T(0);
    }
}

template <typename T>
__global__ void relu_inplace_kernel(T* data, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        T val = data[idx];
        data[idx] = val > T(0) ? val : T(0);
    }
}

// Vectorized kernel for float32 (4 elements per thread)
__global__ void relu_kernel_vec4_float(float* output, const float* input, int64_t numel) {
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < numel) {
        float4 val = reinterpret_cast<const float4*>(input)[idx / 4];
        val.x = val.x > 0.0f ? val.x : 0.0f;
        val.y = val.y > 0.0f ? val.y : 0.0f;
        val.z = val.z > 0.0f ? val.z : 0.0f;
        val.w = val.w > 0.0f ? val.w : 0.0f;
        reinterpret_cast<float4*>(output)[idx / 4] = val;
    } else if (idx < numel) {
        // Handle remaining elements
        for (int64_t i = idx; i < numel; i++) {
            float v = input[i];
            output[i] = v > 0.0f ? v : 0.0f;
        }
    }
}

__global__ void relu_inplace_kernel_vec4_float(float* data, int64_t numel) {
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < numel) {
        float4 val = reinterpret_cast<float4*>(data)[idx / 4];
        val.x = val.x > 0.0f ? val.x : 0.0f;
        val.y = val.y > 0.0f ? val.y : 0.0f;
        val.z = val.z > 0.0f ? val.z : 0.0f;
        val.w = val.w > 0.0f ? val.w : 0.0f;
        reinterpret_cast<float4*>(data)[idx / 4] = val;
    } else if (idx < numel) {
        // Handle remaining elements
        for (int64_t i = idx; i < numel; i++) {
            float v = data[i];
            data[i] = v > 0.0f ? v : 0.0f;
        }
    }
}

// ============ HOST CODE ============

torch::Tensor launch(torch::Tensor input, bool inplace) {
    // 1. Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    
    // Make contiguous if needed
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    
    int64_t numel = input.numel();
    
    // 2. Output tensor creation (or use input for inplace)
    torch::Tensor output;
    if (inplace) {
        output = input;
    } else {
        output = torch::empty_like(input);
    }
    
    // 3. Kernel launch parameters
    // GTX 1660 Ti has Turing architecture with good memory bandwidth
    // Use 256 threads per block for good occupancy
    const int threads = 256;
    
    // 4. Launch kernel based on dtype
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "relu_cuda", ([&] {
        if (input.scalar_type() == torch::kFloat32 && numel >= 1024) {
            // Use vectorized kernel for float32 with sufficient elements
            // Process 4 elements per thread
            int64_t vec_numel = (numel + 3) / 4;
            int blocks = (vec_numel + threads - 1) / threads;
            
            if (inplace) {
                relu_inplace_kernel_vec4_float<<<blocks, threads>>>(
                    output.data_ptr<float>(),
                    numel
                );
            } else {
                relu_kernel_vec4_float<<<blocks, threads>>>(
                    output.data_ptr<float>(),
                    input.data_ptr<float>(),
                    numel
                );
            }
        } else {
            // Use scalar kernel for other types or small tensors
            int blocks = (numel + threads - 1) / threads;
            
            if (inplace) {
                relu_inplace_kernel<scalar_t><<<blocks, threads>>>(
                    output.data_ptr<scalar_t>(),
                    numel
                );
            } else {
                relu_kernel<scalar_t><<<blocks, threads>>>(
                    output.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    numel
                );
            }
        }
    }));
    
    // 5. Error checking
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch error: ", cudaGetErrorString(err));
    
    // 6. Return tensor
    return output;
}

// [END kernel.cu]