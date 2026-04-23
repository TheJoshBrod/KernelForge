"""File to construct and pull prompts."""
import os
import torch

from src.optimizer.benchmarking.profile_entries import (
    descriptor_meta,
    is_recompute_output,
    is_tensor_descriptor,
)


def get_system_prompt() -> str:
    """Returns system prompt for generator

    Returns:
        str: System Prompt
    """

    prompt = ""
    with open("src/generator/prompts/GeneratorSystemPrompt.md") as f:
        prompt = f.read()
    target = _target_device()
    if target == "cpu":
        prompt += """

TARGET DEVICE: CPU

Rules:
- Generate a CPU-only C++ extension (no CUDA headers, no __global__ kernels).
- Do NOT call cuda APIs or use CUDA-specific checks.
- Do NOT require .is_cuda(); tensors are CPU.
- Keep torch::Tensor launch(...) signature unchanged.
"""
    elif target == "mps":
        prompt += """

TARGET DEVICE: MPS (Apple Silicon)

Rules:
- Generate a CPU/MPS-compatible C++ extension (no CUDA headers, no __global__ kernels).
- Do NOT call cuda APIs or use CUDA-specific checks.
- Do NOT require .is_cuda(); tensors are MPS.
- Keep torch::Tensor launch(...) signature unchanged.
"""
    elif target == "cuda":
        prompt += """

TARGET DEVICE: CUDA

Rules:
- Generate a CUDA kernel and CUDA-aware C++ wrapper.
- Enforce .is_cuda() checks for tensor inputs.
"""
    elif target == "triton":
        prompt = ""
        with open("src/generator/prompts/TritonGeneratorSystemPrompt.md") as f:
            prompt = f.read()
        try:
            import triton as _triton
            triton_version = _triton.__version__
        except Exception:
            triton_version = "unknown"
        prompt = f"Target runtime: OpenAI Triton {triton_version}\n\n" + prompt
    return prompt


def _target_device() -> str:
    value = os.environ.get("KFORGE_TARGET_DEVICE", "").strip().lower()
    if value in {"gpu", "cuda"}:
        return "cuda"
    if value in {"triton"}:
        return "triton"
    if value == "mps":
        return "mps"
    if value == "cpu":
        return "cpu"
    return ""


def _merge_dynamic_dims(values_list):
    if not values_list:
        return []
    ref = list(values_list[0])
    dynamic = list(ref)
    for val in values_list[1:]:
        if len(val) != len(ref):
            return "Rank Varies"
        for i, dim in enumerate(val):
            if dim != dynamic[i]:
                dynamic[i] = -1
    return dynamic


def _summarize_scalar(values: list):
    if not values:
        return {}
    uniq = list(dict.fromkeys(values))[:5]
    summary = {"examples": uniq}
    try:
        numeric_vals = [v for v in values if isinstance(v, (int, float))]
        if numeric_vals:
            summary["min"] = min(numeric_vals)
            summary["max"] = max(numeric_vals)
    except Exception:
        pass
    return summary


def _tensor_stats(value: torch.Tensor) -> dict:
    target = _target_device()
    device = target or str(value.device)
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "device": device,
        "contiguous": bool(value.is_contiguous()),
        "requires_grad": bool(value.requires_grad),
        "numel": int(value.numel()),
    }


def _tensor_like_stats(value) -> dict | None:
    if torch.is_tensor(value):
        return _tensor_stats(value)
    if is_tensor_descriptor(value) or is_recompute_output(value):
        meta = dict(descriptor_meta(value))
        if not meta:
            return None
        target = _target_device()
        if target:
            meta["device"] = target
        meta.setdefault("contiguous", False)
        meta.setdefault("requires_grad", False)
        meta["representation"] = "recompute_output" if is_recompute_output(value) else value.get("kind", "tensor_descriptor")
        return meta
    return None


def _summarize_value(value):
    tensor_stats = _tensor_like_stats(value)
    if tensor_stats is not None:
        return tensor_stats
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value),
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": list(value.keys()),
        }
    return {"type": type(value).__name__, "value": value}


def _infer_param_order(call_list):
    for call in call_list:
        sig = call.get("signature", {}) if isinstance(call, dict) else {}
        params = sig.get("params", []) if isinstance(sig, dict) else []
        defaults = sig.get("defaults", {}) if isinstance(sig, dict) else {}
        if params:
            return list(params), defaults
    # Fallback: positional args then kwargs in observed order
    if call_list:
        first = call_list[0]
        args = first.get("args", [])
        kwargs = first.get("kwargs", {})
        param_order = [f"arg{i}" for i in range(len(args))]
        if isinstance(kwargs, dict):
            param_order.extend(list(kwargs.keys()))
        return param_order, {}
    return [], {}


def generate_function_spec_from_calls(call_list, function_name):
    """
    Extract function specification from ALL tracked PyTorch calls.
    Aggregates types and detects dynamic shapes across all iterations.
    """
    if not call_list:
        return None

    # 1. Data Aggregation
    # We will store observed properties for every argument across all calls
    param_stats = {}

    param_order, defaults = _infer_param_order(call_list)

    for call_idx, call in enumerate(call_list):
        # Normalize args to dict keys (arg0, arg1...) to match kwargs
        current_params = {}
        for i, arg in enumerate(call.get("args", [])):
            name = param_order[i] if i < len(param_order) else f"arg{i}"
            current_params[name] = arg
        current_params.update(call.get("kwargs", {}))

        # Update stats for each parameter
        for name, value in current_params.items():
            if name not in param_stats:
                param_stats[name] = {
                    "types": set(),
                    "shapes": [],
                    "strides": [],
                    "dtypes": set(),
                    "devices": set(),
                    "contiguous": set(),
                    "requires_grad": set(),
                    "numel": set(),
                    "list_lens": set(),
                    "scalar_values": [],
                }

            tensor_stats = _tensor_like_stats(value)

            # Record Type
            param_stats[name]["types"].add(torch.Tensor if tensor_stats is not None else type(value))

            # Record Shape (for Tensors and v2 tensor descriptors)
            if tensor_stats is not None:
                target = _target_device()
                device = target or str(tensor_stats.get("device", "unknown"))
                param_stats[name]["shapes"].append(list(tensor_stats.get("shape", [])))
                param_stats[name]["strides"].append(list(tensor_stats.get("stride", [])))
                param_stats[name]["dtypes"].add(str(tensor_stats.get("dtype", "unknown")))
                param_stats[name]["devices"].add(device)
                param_stats[name]["contiguous"].add(bool(tensor_stats.get("contiguous", False)))
                param_stats[name]["requires_grad"].add(bool(tensor_stats.get("requires_grad", False)))
                if tensor_stats.get("numel") is not None:
                    param_stats[name]["numel"].add(int(tensor_stats.get("numel", 0)))

            # Record Length (for Lists/Tuples)
            elif isinstance(value, (list, tuple)):
                param_stats[name]["list_lens"].add(len(value))
                param_stats[name]["scalar_values"].append(value)
            elif value is None:
                param_stats[name]["scalar_values"].append(None)
            else:
                param_stats[name]["scalar_values"].append(value)

    # 2. Build Specification
    param_specs = []

    for name in param_order:
        stats = param_stats.get(name)
        if not stats:
            continue

        types = stats["types"]
        shapes = stats["shapes"]
        strides = stats["strides"]
        dtypes = stats["dtypes"]
        devices = stats["devices"]
        contiguous = stats["contiguous"]
        requires_grad = stats["requires_grad"]
        numel = stats["numel"]

        spec = {
            "name": name,
            "description": ""
        }

        # --- Logic for Tensors ---
        if torch.Tensor in types:
            if _target_device() == "triton":
                spec["type"] = "torch.Tensor"
            else:
                spec["type"] = "torch::Tensor"

            # Check if it is Optional (sometimes None)
            if type(None) in types:
                if _target_device() == "triton":
                    spec["type"] = "torch.Tensor | None"
                else:
                    spec["type"] = "std::optional<torch::Tensor>"
                spec["description"] += "Optional Tensor (handle null/None). "

            # Analyze Shapes for Dynamics
            if not shapes:
                spec["shape"] = "Unknown"
            else:
                final_shape = _merge_dynamic_dims(shapes)
                spec["shape"] = final_shape
                if final_shape == "Rank Varies":
                    spec["description"] += "Input tensor with varying rank. "
                elif isinstance(final_shape, list) and -1 in final_shape:
                    spec["description"] += (
                        f"Input tensor. Dynamic shape: {final_shape} (-1 indicates variable dim). "
                    )
                else:
                    spec["description"] += f"Input tensor. Fixed shape: {final_shape}. "

            if not strides:
                spec["stride"] = "Unknown"
            else:
                spec["stride"] = _merge_dynamic_dims(strides)

            spec["dtype"] = list(dtypes) if dtypes else ["unknown"]
            spec["device"] = list(devices) if devices else ["unknown"]
            spec["contiguous"] = list(contiguous) if contiguous else []
            spec["requires_grad"] = list(requires_grad) if requires_grad else []
            spec["numel"] = list(numel) if numel else []

        # --- Logic for Lists/Tuples ---
        elif list in types or tuple in types:
            spec["type"] = "list" if _target_device() == "triton" else "std::vector"
            lens = stats["list_lens"]
            if len(lens) > 1:
                spec["description"] = f"List/Tuple with varying lengths: {lens}"
            else:
                spec["description"] = f"List/Tuple of length {list(lens)[0]}"
            spec["examples"] = stats["scalar_values"][:3]

        # --- Logic for Scalars ---
        elif int in types:
            spec["type"] = "int" if _target_device() == "triton" else "int64_t"
            spec["description"] = "Integer scalar"
            spec["stats"] = _summarize_scalar(stats["scalar_values"])
        elif float in types:
            spec["type"] = "float" if _target_device() == "triton" else "double"
            spec["description"] = "Float scalar"
            spec["stats"] = _summarize_scalar(stats["scalar_values"])
        elif bool in types:
            spec["type"] = "bool"
            spec["description"] = "Boolean flag"
            spec["stats"] = _summarize_scalar(stats["scalar_values"])
        elif str in types:
            spec["type"] = "str" if _target_device() == "triton" else "std::string"
            spec["description"] = "String parameter"
            spec["stats"] = _summarize_scalar(stats["scalar_values"])
        else:
            spec["type"] = "auto"
            spec["description"] = "Unknown/Complex type"
            spec["stats"] = _summarize_scalar(stats["scalar_values"])

        param_specs.append(spec)

    return {
        "function_name": function_name,
        "num_calls": len(call_list),
        "parameters": param_specs,
        "signature": {
            "params": param_order,
            "defaults": defaults,
        },
    }


def format_operator_prompt(function_spec, profiler_context=None, template: str | None = None):
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

Based on {function_spec['num_calls']} tracked call(s), implement this operator.

Signature parameters (exact order): {function_spec['signature']['params']}
Defaults: {function_spec['signature']['defaults']}

**Parameters:**
"""
    target = _target_device()
    if target:
        prompt += f"\nTarget device: {target}\n"

    # List all parameters
    for i, param in enumerate(function_spec['parameters'], 1):
        prompt += f"\n{i}. `{param['name']}` ({param['type']})"
        if 'shape' in param:
            prompt += f"\n   - Shape: {param['shape']}"
            prompt += f"\n   - Stride: {param.get('stride', 'unknown')}"
            prompt += f"\n   - Dtype(s): {param.get('dtype', 'unknown')}"
            prompt += f"\n   - Device(s): {param.get('device', 'unknown')}"
            prompt += f"\n   - Contiguous: {param.get('contiguous', [])}"
            prompt += f"\n   - Requires grad: {param.get('requires_grad', [])}"
            prompt += f"\n   - Numel: {param.get('numel', [])}"
        elif 'value' in param and param['value'] is not None:
            prompt += f"\n   - Default/Example: {param['value']}"
        elif 'stats' in param:
            prompt += f"\n   - Stats: {param['stats']}"
        prompt += f"\n   - {param['description']}"

    prompt += """

### Output Specification
"""
    output_spec = function_spec.get("output", {})
    if output_spec:
        for k, v in output_spec.items():
            prompt += f"\n- {k}: {v}"
    else:
        prompt += "\n- Output: not available"

    examples = function_spec.get("examples", [])
    if examples:
        prompt += "\n\n### Example Calls (summarized)\n"
        for idx, ex in enumerate(examples, 1):
            prompt += f"\nExample {idx}:\n"
            prompt += f"- args: {ex.get('args')}\n"
            prompt += f"- kwargs: {ex.get('kwargs')}\n"
            prompt += f"- output: {ex.get('output')}\n"

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

    if template:
        prompt += """

### Reference Kernel Template (use as a starting point)
You may reuse structure and helper functions from this template, but ensure the signature
matches the parameters above and the kernel is correct for this operator.

```cpp
"""
        prompt += template
        prompt += "\n```\n"

    # Add implementation guidance — backend-specific
    if _target_device() == "triton":
        prompt += """
### Implementation Requirements

1. **Python Wrapper Function (`launch`):**
   - Must accept ALL parameters listed above in the exact order
   - Optional parameters (None) should be handled with conditional logic
   - Return type must match the output specification

2. **Triton Kernel (`@triton.jit`):**
   - Must handle all tensor inputs via pointer arithmetic
   - Use `tl.load`/`tl.store` with proper masking
   - Block sizes should be `tl.constexpr` parameters

3. **Parameter Handling:**
   - Validate tensor inputs (`.is_cuda`, `.contiguous()`)
   - Pass scalar parameters to kernel directly
   - Handle optional tensors (check for None)

4. **Output Format:**
   - You MUST wrap the complete kernel.py file contents in special tags.
   - Start with `# [START kernel.py]`
   - End with `# [END kernel.py]`
   - Do not include any other text inside these tags.

Now generate the complete kernel.py file following the system prompt guidelines.
"""
    else:
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

4. **Output Format:**
   - You MUST wrap the complete kernel.cu file contents in special tags.
   - Start with `// [START kernel.cu]`
   - End with `// [END kernel.cu]`
   - Do not include any other text inside these tags.

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


def generate_full_llm_prompt(calls_list, function_name, profiler_output=None, template: str | None = None):
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
    # Attach output spec if present
    output_spec = None
    try:
        for call in calls_list:
            out = call.get("output")
            if torch.is_tensor(out):
                output_spec = _tensor_stats(out)
                break
            if isinstance(out, (list, tuple)):
                output_spec = {"type": type(out).__name__, "length": len(out)}
                break
    except Exception:
        output_spec = None
    if output_spec:
        spec["output"] = output_spec

    examples = []
    for call in calls_list[:2]:
        try:
            examples.append(
                {
                    "args": [_summarize_value(v) for v in call.get("args", [])],
                    "kwargs": {
                        k: _summarize_value(v)
                        for k, v in call.get("kwargs", {}).items()
                    },
                    "output": _summarize_value(call.get("output")),
                }
            )
        except Exception:
            continue
    if examples:
        spec["examples"] = examples

    full_prompt = format_operator_prompt(spec, profiler_context, template=template)

    return full_prompt


def get_repair_prompt(function_name: str, attempt: int, feedback: str) -> str:
    if _target_device() == "triton":
        return f"""The previous kernel for {function_name} failed validation on attempt {attempt + 1}.

Error/feedback:
{feedback}

Repair instructions:
- Keep the launch() signature EXACTLY the same — do NOT change argument order or types.
- Only modify the @triton.jit kernel body.
- FORBIDDEN in kernel launch: `.data_ptr()` — NEVER pass `tensor.data_ptr()` to the kernel.
  Pass the tensor object directly. Triton infers the pointer type from the tensor's dtype.
  Passing a raw int64 from .data_ptr() causes "Unsupported ptr type" errors.
- FORBIDDEN as type annotations: `tl.pointer`, `tl.float32_ptr`, `tl.uint64` — none exist.
  Do NOT annotate pointer parameters. Only annotate `tl.constexpr` for compile-time constants.
- FORBIDDEN: `tl.broadcast(tensor, (A, B))` with a tuple shape — causes `'tuple_type' has no
  attribute 'is_block'`. Use `t[:, None]` and `t[None, :]` for 2D broadcasting instead.
- FORBIDDEN inside @triton.jit: `continue`, `break`, `while`, `tl.any()`, `tl.all()`,
  data-dependent `if` over tensor values, `try`/`except`.
- For early-exit patterns: remove them entirely and use masked operations instead.
  tl.store with mask=False is a no-op — always use masking, never `continue`.
- For any/all checks: use `tl.reduce_or(mask)` (returns scalar bool) instead of
  `tl.any(...)`, or restructure to avoid the check entirely using tl.where.
- For reductions over masked elements: use `tl.sum(tl.where(mask, val, 0.0))`.
- Ensure all tensor outputs match PyTorch numerically and in shape.
""".strip()
    return f"""
The previous kernel for {function_name} failed validation on attempt {attempt + 1}.

ERROR SUMMARY (do not ignore):
{feedback}

Repair instructions:
- Keep the launch() signature EXACTLY the same.
- Do NOT change argument order or types.
- Only modify CUDA kernel logic, indexing, dtype handling, and launch configuration.
- Do NOT add PYBIND11_MODULE blocks.
- Ensure all tensor outputs match PyTorch numerically and in shape.
""".strip()
