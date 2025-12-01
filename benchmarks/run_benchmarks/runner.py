import torch
import torch.autograd.profiler as profiler
import torch.nn.functional as F
from functools import wraps
from PIL import Image
import os
import glob
import inspect
import pyarrow.parquet as pq
from pathlib import Path
import importlib.util
import sys
import time

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ****************************
# Global Configuration & Setup
# ****************************

run_compiled = False
for arg in sys.argv:
    if arg == "--compiled":
        print("true")
        run_compiled = True 

CUSTOM_KERNELS = {}

compiled_root = Path("benchmarks/run_benchmarks/compiled_kernels")

print("Loading compiled kernels...")
if compiled_root.exists():
    for kernel_dir in compiled_root.iterdir():
        if not kernel_dir.is_dir():
            continue
        
        so_files = list(kernel_dir.glob("*.so"))
        if not so_files:
            continue
        
        kernel_name = kernel_dir.name
        so_file = so_files[0]
        
        try:
            module_name = so_file.stem
            spec = importlib.util.spec_from_file_location(module_name, so_file)
            if spec is None or spec.loader is None:
                print(f"✗ Could not load {kernel_name}")
                continue
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            if kernel_name.startswith("torch_nn_functional_"):
                func_name = kernel_name.replace("torch_nn_functional_", "")
                key = f"torch.nn.functional.{func_name}"
            else:
                key = f"torch.nn.functional.{kernel_name}"
            
            CUSTOM_KERNELS[key] = module.launch
            
            print(f"✓ Loaded: {key}")
            
        except Exception as e:
            print(f"✗ Failed to load {kernel_name}: {e}")
else:
    print(f"⚠ Compiled kernels directory not found: {compiled_root}")

print(f"\nTotal custom kernels loaded: {len(CUSTOM_KERNELS)}")
print(f"Available keys: {list(CUSTOM_KERNELS.keys())}\n")

# ***************************************************
# Track specific PyTorch Calls with CUDA Event Timing
# ***************************************************
_wrapped = set()
ENABLE_WRAPPING = True
CALL_STATS = {}  # Track how many times each kernel is called
TIMING_STATS = {}  # Track execution times with CUDA events


# *******************************
# CUDA Utilities & Timing Helpers
# *******************************

def move_to_cuda(item):
    """Recursively move tensors to CUDA and make them contiguous."""
    if torch.is_tensor(item):
        item = item.cuda() if not item.is_cuda else item
        return item.contiguous()
    elif isinstance(item, (list, tuple)):
        return type(item)(move_to_cuda(x) for x in item)
    elif isinstance(item, dict):
        return {k: move_to_cuda(v) for k, v in item.items()}
    return item

def normalize_args_kwargs(args: list, kwargs: dict, params: list, defaults: dict) -> tuple[list, dict]:
    """
    Normalize args and kwargs into a complete positional argument list.
    """
    if not params:
        return args, kwargs

    normalized = list(args)
    remaining_kwargs = dict(kwargs)
    
    for i in range(len(normalized), len(params)):
        param_name = params[i]
        
        if param_name in remaining_kwargs:
            normalized.append(remaining_kwargs.pop(param_name))
        elif param_name in defaults:
            normalized.append(defaults[param_name])
        else:
            break
    
    return normalized, remaining_kwargs

def time_kernel_execution(kernel_fn, *args, num_warmup=3):
    """
    Time kernel execution using CUDA events for precise GPU timing.
    
    Args:
        kernel_fn: The kernel function to execute
        *args: Arguments to pass to the kernel
        num_warmup: Number of warmup runs (default: 3)
    
    Returns:
        tuple: (result, execution_time_ms)
    """
    # Warmup runs
    for _ in range(num_warmup):
        _ = kernel_fn(*args)
    
    # Synchronize before timing
    torch.cuda.synchronize()
    
    # Create CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Time the execution
    start_event.record()
    result = kernel_fn(*args)
    end_event.record()
    
    # Wait for completion
    torch.cuda.synchronize()
    
    # Get elapsed time in milliseconds
    elapsed_time = start_event.elapsed_time(end_event)
    
    return result, elapsed_time


# *******************************************
# Dynamic Function Wrapping & Monkey Patching
# *******************************************

def wrap_function(module, func_name):
    if not ENABLE_WRAPPING:
        return
    func = getattr(module, func_name)
    if func in _wrapped:
        return
    _wrapped.add(func)

    module_path = module.__name__
    key = f"{module_path}.{func_name}"
    
    # Get function signature to extract parameter order and defaults
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        defaults = {}
        for param_name, param in sig.parameters.items():
            if param.default != inspect.Parameter.empty:
                defaults[param_name] = param.default
    except (ValueError, TypeError):
        sig = None
        params = []
        defaults = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if we have a custom kernel for this function
        if key in CUSTOM_KERNELS:
            # Track usage
            if key not in CALL_STATS:
                CALL_STATS[key] = 0
                TIMING_STATS[key] = []
            CALL_STATS[key] += 1
            
            try:
                # Normalize args and kwargs to positional arguments
                normalized_args, remaining_kwargs = normalize_args_kwargs(args, kwargs, params, defaults)
                
                # Move tensors to CUDA AND make them contiguous
                cuda_args = []
                for item in normalized_args:
                    if torch.is_tensor(item):
                        item = item.cuda() if not item.is_cuda else item
                        item = item.contiguous()
                        cuda_args.append(item)
                    else:
                        cuda_args.append(move_to_cuda(item))
                
                # Time the custom kernel execution using CUDA events
                if run_compiled:
                    model = CUSTOM_KERNELS[key]
                else:
                    model = func
                result, elapsed_time = time_kernel_execution(model, *cuda_args)
                
                # Store timing information
                TIMING_STATS[key].append(elapsed_time)
                
                return result
                
            except Exception as e:
                # Fallback to original function if custom kernel fails
                print(f"⚠ Custom kernel {key} failed, falling back to PyTorch: {e}")
                return func(*args, **kwargs)

        # Use original PyTorch function
        return func(*args, **kwargs)

    setattr(module, func_name, wrapper)

def selective_wrap():
    """Only wrap functions that have custom implementations."""
    print("Wrapping compiled functions...")
    wrapped_count = 0
    for kernel_key in CUSTOM_KERNELS.keys():
        if kernel_key.startswith("torch.nn.functional."):
            func_name = kernel_key.split(".")[-1]
            if hasattr(F, func_name):
                wrap_function(F, func_name)
                wrapped_count += 1
    
    print(f"Wrapped {wrapped_count} functions with custom kernels")


# ********************
# Run model Functions
# ********************

def profile_image_model(model_name: str = "facebook/convnext-tiny-224"):
    """Runs Image Classification model from HuggingFace's Transformer library."""
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device_str)
 
    image_paths = glob.glob("benchmarks/generate_benchmarks/images/*.JPEG")
    for path in image_paths[:7]:
        if os.path.exists(path):
            image = Image.open(path)
            inputs = processor(images=image, return_tensors="pt")
            config = AutoConfig.from_pretrained(model_name)
            id2label = config.id2label
            
            inputs = {k: v.to(device_str) for k, v in inputs.items()}
            
            with profiler.profile(record_shapes=True, use_device=device_str) as prof:
                with profiler.record_function("forward"):
                    with torch.no_grad():
                        outputs = model(**inputs)

            logits = outputs.logits
            predicted_class_id = logits.argmax(-1).item()
            label = id2label[predicted_class_id]

def profile_text_model(model_name: str = "bert-base-uncased"):
    """Runs a HuggingFace text classification model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    pq_path = "benchmarks/generate_benchmarks/text_inputs.parquet"
    table = pq.read_table(pq_path)
    df = table.to_pandas()
    text_samples = df["sentence"].tolist()[:20]

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device_str)

    for text in text_samples:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device_str) for k, v in inputs.items()}

        with profiler.profile(record_shapes=True, use_device=device_str) as prof:
            with profiler.record_function("forward"):
                with torch.no_grad():
                    outputs = model(**inputs)

        logits = outputs.logits
        predicted_class = torch.argmax(logits).item()

        config = AutoConfig.from_pretrained(model_name)
        label_map = config.id2label if hasattr(config, "id2label") else None
        label_text = label_map[predicted_class] if label_map else predicted_class


# **********************
# Statistics & Reporting
# **********************

def print_kernel_stats():
    """Print statistics about custom kernel usage and timing."""
    print("\n" + "="*80)
    print("CUSTOM KERNEL USAGE AND TIMING STATISTICS")
    print("="*80)
    
    if CALL_STATS:
        print(f"\n{'Kernel Name':<50} {'Calls':<8} {'Avg (ms)':<12} {'Total (ms)':<12}")
        print("-" * 80)
        
        for func_name in sorted(CALL_STATS.keys()):
            count = CALL_STATS[func_name]
            timings = TIMING_STATS.get(func_name, [])
            
            if timings:
                avg_time = sum(timings) / len(timings)
                total_time = sum(timings)
                print(f"{func_name:<50} {count:<8} {avg_time:<12.4f} {total_time:<12.4f}")
            else:
                print(f"{func_name:<50} {count:<8} {'N/A':<12} {'N/A':<12}")
        
        # Print summary
        total_calls = sum(CALL_STATS.values())
        all_timings = [t for times in TIMING_STATS.values() for t in times]
        if all_timings:
            total_gpu_time = sum(all_timings)
            print("-" * 80)
            print(f"{'TOTAL':<50} {total_calls:<8} {'':<12} {total_gpu_time:<12.4f}")
            print(f"\nTotal GPU time (custom kernels): {total_gpu_time:.4f} ms")
    else:
        print("  No custom kernels were called.")
    
    print("="*80 + "\n")


# **************************
# Main Execution Entry Point
# **************************

def main():
    total_time = 0

    # Wraps functions we compiled
    selective_wrap()

    for i in range(11):
        start = time.time()
        
        image_model_names = [
            "microsoft/resnet-50",
            "microsoft/swin-base-patch4-window7-224",
            "facebook/convnext-base-224"
        ]
        for model_name in image_model_names:
            profile_image_model(model_name)

        text_classification_model_names = [
            "distilbert-base-uncased-finetuned-sst-2-english"
        ]
        for model_name in text_classification_model_names:
            profile_text_model(model_name)
        
        end = time.time()

        # Skip warmup iteration
        if i == 0:
            continue
        total_time += end - start
    
    print(f"\nTotal wall-clock time (excluding warmup): {total_time:.4f} seconds")
    
    # Print detailed kernel statistics
    print_kernel_stats()

    
if __name__ == "__main__":
    main()