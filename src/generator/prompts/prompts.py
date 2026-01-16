"""File to construct and pull prompts."""
import torch


def get_system_prompt() -> str:
    """Returns system prompt for generator

    Returns:
        str: System Prompt
    """

    prompt = ""
    with open("src/generator/prompts/GeneratorSystemPrompt.md") as f:
        prompt = f.read()
    return prompt


def generate_function_spec_from_calls(call_list, function_name):
    """
    Extract function specification from ALL tracked PyTorch calls.
    Aggregates types and detects dynamic shapes across all iterations.
    """
    if not call_list:
        return None

    # 1. Data Aggregation
    # We will store observed properties for every argument across all calls
    # param_stats = { "arg0": { "types": set(), "shapes": [], "is_list": False }, ... }
    param_stats = {}

    param_order = []  # To keep arguments in correct order

    for call_idx, call in enumerate(call_list):
        # Normalize args to dict keys (arg0, arg1...) to match kwargs
        current_params = {}
        for i, arg in enumerate(call.get("args", [])):
            current_params[f"arg{i}"] = arg
        current_params.update(call.get("kwargs", {}))

        # Initialize param order from the first call
        if call_idx == 0:
            param_order = list(current_params.keys())

        # Update stats for each parameter
        for name, value in current_params.items():
            if name not in param_stats:
                param_stats[name] = {
                    "types": set(), "shapes": [], "list_lens": set()}

            # Record Type
            param_stats[name]["types"].add(type(value))

            # Record Shape (for Tensors)
            if isinstance(value, torch.Tensor):
                param_stats[name]["shapes"].append(list(value.shape))

            # Record Length (for Lists/Tuples)
            elif isinstance(value, (list, tuple)):
                param_stats[name]["list_lens"].add(len(value))

    # 2. Build Specification
    param_specs = []

    for name in param_order:
        stats = param_stats.get(name)
        if not stats:
            continue

        types = stats["types"]
        shapes = stats["shapes"]

        spec = {
            "name": name,
            "description": ""
        }

        # --- Logic for Tensors ---
        if torch.Tensor in types:
            spec["type"] = "torch::Tensor"

            # Check if it is Optional (sometimes None)
            if type(None) in types:
                spec["type"] = "std::optional<torch::Tensor>"
                spec["description"] += "Optional Tensor (handle null/None). "

            # Analyze Shapes for Dynamics
            if not shapes:
                # Can happen if Tensor type was seen but only as None (rare edge case)
                spec["shape"] = "Unknown"
            else:
                # Compare all observed shapes to find dynamic dimensions
                # Start with the first observed shape as reference
                ref_shape = shapes[0]
                final_shape = list(ref_shape)

                is_dynamic_rank = False

                for s in shapes[1:]:
                    if len(s) != len(ref_shape):
                        final_shape = "Rank Varies"
                        is_dynamic_rank = True
                        break

                    for dim_i, dim_val in enumerate(s):
                        if dim_val != final_shape[dim_i]:
                            final_shape[dim_i] = -1  # Mark as dynamic

                spec["shape"] = final_shape

                # Format description
                if is_dynamic_rank:
                    spec["description"] += "Input tensor with varying rank. "
                elif -1 in final_shape:
                    spec["description"] += f"Input tensor. Dynamic shape: {final_shape} (-1 indicates variable dim). "
                else:
                    spec["description"] += f"Input tensor. Fixed shape: {final_shape}. "

            # Grab dtype from the last non-None value seen
            # (In a real implementation, you might check if dtypes vary too)
            spec["dtype"] = "mixed"  # Placeholder, usually consistent

        # --- Logic for Lists/Tuples ---
        elif list in types or tuple in types:
            spec["type"] = "std::vector"
            lens = stats["list_lens"]
            if len(lens) > 1:
                spec["description"] = f"List/Tuple with varying lengths: {lens}"
            else:
                spec["description"] = f"List/Tuple of length {list(lens)[0]}"

        # --- Logic for Scalars ---
        elif int in types:
            spec["type"] = "int64_t"
            spec["description"] = "Integer scalar"
        elif float in types:
            spec["type"] = "double"
            spec["description"] = "Float scalar"
        elif bool in types:
            spec["type"] = "bool"
            spec["description"] = "Boolean flag"
        elif str in types:
            spec["type"] = "std::string"
            spec["description"] = "String parameter"
        else:
            spec["type"] = "auto"
            spec["description"] = "Unknown/Complex type"

        param_specs.append(spec)

    return {
        "function_name": function_name,
        "num_calls": len(call_list),
        "parameters": param_specs
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


def generate_full_llm_prompt(calls_list, function_name, profiler_output=None):
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
    spec = generate_function_spec_from_calls(calls_list, function_name)
    if spec is None:
        return f"Error: Could not generate spec for {function_name}"

    # Parse profiler output if provided
    profiler_context = None
    if profiler_output:
        profiler_context = parse_profiler_output(profiler_output)

    # Combine system prompt + operator specification
    full_prompt = format_operator_prompt(spec, profiler_context)

    return full_prompt
