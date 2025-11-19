import torch
import json 
import re

aten_to_cuda = """
SYSTEM PROMPT — CUDA PyTorch Extension Generator
-------------------------------------------------

Your job is to generate a **single compilable CUDA PyTorch extension source file** named `kernel.cu`.

-----------------------------------------------
 OUTPUT RULES (CRITICAL — NEVER BREAK)
-----------------------------------------------
1. Output **exactly ONE code block**.
2. The code block must start with:

```
cpp
// [START kernel.cu]
```

   and end with:

```
cpp
// [END kernel.cu]
```

3. The code must contain:
   - CUDA kernel(s) (device code) defined **before** PyTorch includes.
   - C++ host wrapper function(s).
   - A Pybind11 module exposing a function named **launch**.

4. No text, explanation, or comments outside the code block.

-----------------------------------------------
 REQUIRED CODE STRUCTURE
-----------------------------------------------

```
cpp
// [START kernel.cu]
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

// ============ DEVICE CODE (CUDA kernels only) ============
template <typename T>
__global__ void my_kernel(...) {
    // ...
}

// ============ HOST CODE ============
#include <torch/extension.h>

torch::Tensor launch_my_op(...) {
    // Input validation
    // Contiguous conversion
    // Type dispatch (float32, float64, half)
    // Kernel launch
    // Error checking
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch_my_op, "Description");
}
// [END kernel.cu]
```

-----------------------------------------------
 REQUIRED BEHAVIOR RULES
-----------------------------------------------
- Always use `int64_t` for shapes, indices, sizes.
- Always call `.contiguous()` on tensor inputs.
- Always validate tensor arguments (`is_cuda()`, dtype).
- Always check kernel failures with `cudaGetLastError()`.
- Always support dtype dispatch for: float32, float64, half (c10::Half).
- Use `std::vector<int64_t>` for shape construction.
- Never allocate GPU memory manually (`cudaMalloc`, etc.)
- Never include `main()`, logging, prints, or extra blocks.

-----------------------------------------------
Input: Operator description
Output: Complete valid `kernel.cu` implementation following rules
-----------------------------------------------
"""


def generate_function_spec_from_calls(calls_dict, function_name):
    """
    Extract function specification from tracked PyTorch calls.
    Updated to support call records with 'args' and 'kwargs' instead of 'params'.
    """
    import torch
    
    if function_name not in calls_dict:
        return None
    
    call_list = calls_dict[function_name]
    if not call_list:
        return None
    
    ref_call = call_list[0]
    
    # Build combined parameter dict
    params = {}
    
    # Convert args to synthetic param names: arg0, arg1, ...
    for i, arg in enumerate(ref_call.get("args", [])):
        params[f"arg{i}"] = arg
    
    # Add kwargs normally
    params.update(ref_call.get("kwargs", {}))
    
    output = ref_call.get("output", None)
    
    param_specs = []
    for param_name, param_value in params.items():
        if isinstance(param_value, torch.Tensor):
            param_specs.append({
                "name": param_name,
                "type": "torch::Tensor",
                "dtype": str(param_value.dtype).replace('torch.', ''),
                "shape": list(param_value.shape),
                "description": f"Input tensor of shape {list(param_value.shape)}"
            })
        elif param_value is None:
            param_specs.append({
                "name": param_name,
                "type": "optional",
                "value": "None",
                "description": "Optional parameter (default: None)"
            })
        elif isinstance(param_value, bool):
            param_specs.append({
                "name": param_name,
                "type": "bool",
                "value": param_value,
                "description": "Boolean flag"
            })
        elif isinstance(param_value, int):
            param_specs.append({
                "name": param_name,
                "type": "int64_t",
                "value": param_value,
                "description": "Integer parameter"
            })
        elif isinstance(param_value, float):
            param_specs.append({
                "name": param_name,
                "type": "double",
                "value": param_value,
                "description": "Float parameter"
            })
        elif isinstance(param_value, str):
            param_specs.append({
                "name": param_name,
                "type": "std::string",
                "value": f'"{param_value}"',
                "description": "String parameter"
            })
        elif isinstance(param_value, (list, tuple)):
            param_specs.append({
                "name": param_name,
                "type": "std::vector<int64_t>",
                "value": list(param_value),
                "description": "List/tuple parameter"
            })
    
    # Output spec
    if isinstance(output, torch.Tensor):
        output_spec = {
            "type": "torch::Tensor",
            "dtype": str(output.dtype).replace('torch.', ''),
            "shape": list(output.shape),
            "description": f"Output tensor of shape {list(output.shape)}"
        }
    else:
        output_spec = {
            "type": str(type(output).__name__),
            "description": "Non-tensor output"
        }
    
    return {
        "function_name": function_name,
        "num_calls": len(call_list),
        "parameters": param_specs,
        "output": output_spec
    }


def format_operator_prompt(function_spec, profiler_context=None):
    """
    Format the function specification into a clear prompt for the LLM.
    This is what you append to the system prompt.
    
    Args:
        function_spec: Function specification dict
        profiler_context: Optional dict with {'aten_ops': [...], 'cuda_kernels': [...]}
    """
    
    prompt = f"""
## OPERATOR TO IMPLEMENT: {function_spec['function_name']}

### Function Signature

Based on {function_spec['num_calls']} tracked call(s), implement this operator:

**Parameters:**
"""
    
    # List all parameters
    for i, param in enumerate(function_spec['parameters'], 1):
        prompt += f"\n{i}. `{param['name']}` ({param['type']})"
        if 'shape' in param:
            prompt += f"\n   - Shape: {param['shape']}"
            prompt += f"\n   - dtype: {param['dtype']}"
        elif 'value' in param and param['value'] is not None:
            prompt += f"\n   - Default/Example: {param['value']}"
        prompt += f"\n   - {param['description']}"
    
    prompt += f"""

**Returns:**
- Type: {function_spec['output']['type']}
"""
    if 'shape' in function_spec['output']:
        prompt += f"- Shape: {function_spec['output']['shape']}\n"
        prompt += f"- dtype: {function_spec['output']['dtype']}\n"
    
    # Add profiler context if available
    if profiler_context:
        prompt += """

### Execution Context (from PyTorch Profiler)

This shows how PyTorch implements this operation internally:

"""
        if 'aten_ops' in profiler_context and profiler_context['aten_ops']:
            prompt += "**ATen Operations Called:**\n"
            for op in profiler_context['aten_ops']:
                prompt += f"- {op}\n"
            prompt += "\n"
        
        if 'cuda_kernels' in profiler_context and profiler_context['cuda_kernels']:
            prompt += "**CUDA Kernels Launched:**\n"
            for kernel in profiler_context['cuda_kernels']:
                prompt += f"- {kernel}\n"
            prompt += "\n"
        
        prompt += """**What This Means:**
- You may see setup operations (cudaGetDeviceCount, etc.) - ignore these
- Focus on the actual computation kernels
- Your implementation should replicate the core operation's behavior
- You don't need to match PyTorch's internal implementation exactly

"""
    
    # Add implementation guidance
    prompt += """
### Implementation Requirements

1. **C++ Wrapper Function Signature:**
   - Must accept ALL parameters listed above in the exact order
   - Optional parameters (None) should be handled with conditional logic
   - Return type must match the output specification

2. **CUDA Kernel:**
   - Must handle all tensor inputs
   - Compute the operation based on the function semantics
   - Support float32, float64, and half precision

3. **Parameter Handling:**
   - Validate tensor inputs (.is_cuda(), .contiguous())
   - Use scalar parameters directly in kernel launch
   - Handle optional tensors (check for null/validity)

Now generate the complete kernel.cu file following the system prompt guidelines.
"""
    
    return prompt


def parse_profiler_output(profiler_text):
    """
    Parse the profiler output string to extract ATen ops and CUDA kernels.
    
    Args:
        profiler_text: String containing profiler output with [Op: ...] and [Kernel: ...] lines
    
    Returns:
        dict with 'aten_ops' and 'cuda_kernels' lists
    """
    import re
    
    aten_ops = []
    cuda_kernels = []
    
    # Extract [Op: aten::something]
    op_pattern = r'\[Op:\s*(aten::\w+)\]'
    for match in re.finditer(op_pattern, profiler_text):
        aten_ops.append(match.group(1))
    
    # Extract [Kernel: something]
    kernel_pattern = r'\[Kernel:\s*([^\]]+)\]'
    for match in re.finditer(kernel_pattern, profiler_text):
        kernel_name = match.group(1).strip()
        # Filter out setup/instrumentation kernels
        if kernel_name not in ['Activity Buffer Request', 'Instrumentation', 'Resource']:
            cuda_kernels.append(kernel_name)
    
    return {
        'aten_ops': aten_ops,
        'cuda_kernels': cuda_kernels
    }


def generate_full_llm_prompt(calls_dict, function_name, profiler_output=None):
    """
    Complete pipeline: Generate the full prompt to send to an LLM.
    
    Args:
        calls_dict: Tracked function calls
        function_name: Function to implement (e.g., "torch.nn.functional.linear")
        profiler_output: Optional string output from PyTorch profiler
    
    Usage:
        calls = torch.load("tracked_calls.pt")
        
        # Without profiler context
        prompt = generate_full_llm_prompt(calls, "torch.nn.functional.linear")
        
        # With profiler context
        profiler_text = "..."  # Your aten/kernel output
        prompt = generate_full_llm_prompt(calls, "torch.nn.functional.linear", profiler_text)
    """
    
    # Extract function specification
    spec = generate_function_spec_from_calls(calls_dict, function_name)
    if spec is None:
        return f"Error: Could not generate spec for {function_name}"
    
    # Parse profiler output if provided
    profiler_context = None
    if profiler_output:
        profiler_context = parse_profiler_output(profiler_output)
    
    # Combine system prompt + operator specification
    full_prompt = format_operator_prompt(spec, profiler_context)
    
    return full_prompt
