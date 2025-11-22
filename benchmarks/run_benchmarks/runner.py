import torch
import torch.autograd.profiler as profiler
import torch.nn.functional as F
from functools import wraps
from PIL import Image
import os
import glob
import json
import tqdm
import inspect
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
import importlib.util
import sys
import time 

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SKIP_FUNCTIONS = [
    "has_torch_function",
    "handle_torch_function",
    "is_storage",
    "result_type",
    "get_default_dtype",
    "tensor",  # Don't wrap tensor creation
    "as_tensor",
    "from_numpy",
]

# ****************************
# Load Custom Kernels into Dict
# ****************************
CUSTOM_KERNELS = {}

compiled_root = Path("benchmarks/run_benchmarks/compiled_kernels")

print("Loading compiled kernels...")
if compiled_root.exists():
    for kernel_dir in compiled_root.iterdir():
        if not kernel_dir.is_dir():
            continue
        
        # Find .so files in this directory
        so_files = list(kernel_dir.glob("*.so"))
        print(so_files)
        if not so_files:
            continue
        
        kernel_name = kernel_dir.name
        so_file = so_files[0]
        
        try:
            # Load the module
            module_name = so_file.stem
            spec = importlib.util.spec_from_file_location(module_name, so_file)
            if spec is None or spec.loader is None:
                print(f"✗ Could not load {kernel_name}")
                continue
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Add to dict with function name as key
            # Assuming the kernel has a 'launch' function
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

# ****************************
# Track specific PyTorch Calls
# ****************************
_wrapped = set()
ENABLE_WRAPPING = True
CALL_STATS = {}  # Track how many times each kernel is called

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
    
    Args:
        args: Positional arguments as captured
        kwargs: Keyword arguments as captured
        params: List of parameter names in order
        defaults: Dict of default values
    
    Returns:
        tuple: (normalized_args, remaining_kwargs)
    """
    if not params:
        return args, kwargs

    # Start with provided positional args
    normalized = list(args)
    remaining_kwargs = dict(kwargs)
    
    # Fill in any kwargs that should be positional
    for i in range(len(normalized), len(params)):
        param_name = params[i]
        
        if param_name in remaining_kwargs:
            # Use the provided kwarg
            normalized.append(remaining_kwargs.pop(param_name))
        elif param_name in defaults:
            # Use the default value
            normalized.append(defaults[param_name])
        else:
            # No value available
            break
    
    return normalized, remaining_kwargs


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
            CALL_STATS[key] += 1
            
            try:
                # Normalize args and kwargs to positional arguments
                normalized_args, remaining_kwargs = normalize_args_kwargs(args, kwargs, params, defaults)
                
                # Move tensors to CUDA AND make them contiguous
                cuda_args = []
                for item in normalized_args:
                    if torch.is_tensor(item):
                        item = item.cuda() if not item.is_cuda else item
                        item = item.contiguous()  # ← ADD THIS
                        cuda_args.append(item)
                    else:
                        cuda_args.append(move_to_cuda(item))
                
                # Call custom kernel with positional args only
                result = CUSTOM_KERNELS[key](*cuda_args)
                
                # Ensure CUDA operations complete
                if torch.is_tensor(result):
                    torch.cuda.synchronize()
                
                return result
                
            except Exception as e:
                # Fallback to original function if custom kernel fails
                print(f"⚠ Custom kernel {key} failed, falling back to PyTorch: {e}")
                return func(*args, **kwargs)

        # Use original PyTorch function
        return func(*args, **kwargs)

    setattr(module, func_name, wrapper)

# Wrap all functions in torch.nn.functional AND torch module
print("Wrapping torch.nn.functional functions...")
for name in dir(F):
    if name.startswith("_"):
        continue  # skip private names like _in_projection, etc.
    if name in SKIP_FUNCTIONS or any(skip in name for skip in ["torch_function", "storage", "result_type", "dtype"]):
        continue  # skip helpers by substring match
    obj = getattr(F, name)
    if callable(obj):
        wrap_function(F, name)

def profile_image_model(model_name: str = "facebook/convnext-tiny-224"):
    """Runs Image Classification model from HuggingFace's Transformer library to profile input and outputs. 
    Outputs the input/output pairs to pt file separate by PyTorch function name

    Args:
        model_name (str, optional): Name of Hugging Face image classification model. Defaults to "facebook/convnext-tiny-224".
    """

    # ****************************
    #    Initialize PyTorch Model
    # ****************************

    # print(f"Loading {model_name}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device_str)
 
    # ****************************
    #        Run Model 
    # ****************************

    # print(f"Running {model_name}...")
    image_paths = glob.glob("benchmarks/generate_benchmarks/images/*.JPEG")
    for path in image_paths[:7]:
        if os.path.exists(path):
            # Open image
            image = Image.open(path)
            
            # Configure inputs
            inputs = processor(images=image, return_tensors="pt")
            config = AutoConfig.from_pretrained(model_name)
            id2label = config.id2label
            
            inputs = {k: v.to(device_str) for k, v in inputs.items()}
            
            # Run model
            with profiler.profile(record_shapes=True, use_device=device_str) as prof:
                with profiler.record_function("forward"):
                    with torch.no_grad():
                        outputs = model(**inputs)

            # Calculate Output 
            logits = outputs.logits
            predicted_class_id = logits.argmax(-1).item()
            label = id2label[predicted_class_id]
            # print(label)


def profile_text_model(model_name: str = "bert-base-uncased"):
    """
    Runs a HuggingFace text classification model using text samples from a JSON file
    and profiles PyTorch forward passes. Saves profiling input/output to .pt files
    grouped by PyTorch function names.

    Expected JSON format:
        {
            "samples": [
                "This movie was awesome!",
                "I hate the food here.",
                "Weather looks great today."
            ]
        }

    Args:
        model_name (str): Hugging Face model name.
        input_json (str): Path to JSON file containing text samples.
    """

    # print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Load text samples    
    pq_path = "benchmarks/generate_benchmarks/text_inputs.parquet"
    table = pq.read_table(pq_path)
    df = table.to_pandas()
    text_samples = df["sentence"].tolist()[:20]

    # print(f"Running {model_name}...")
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device_str)

    for text in text_samples:
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device_str) for k, v in inputs.items()}

        # Profile model
        with profiler.profile(record_shapes=True, use_device=device_str) as prof:
            with profiler.record_function("forward"):
                with torch.no_grad():
                    outputs = model(**inputs)

        # Get predicted label
        logits = outputs.logits
        predicted_class = torch.argmax(logits).item()

        # Try mapping to readable class labels if available
        config = AutoConfig.from_pretrained(model_name)
        label_map = config.id2label if hasattr(config, "id2label") else None
        label_text = label_map[predicted_class] if label_map else predicted_class

        # print(f"TEXT: {text}\nPREDICTION: {label_text}\n")


def print_kernel_stats():
    """Print statistics about custom kernel usage."""
    print("\n" + "="*60)
    print("CUSTOM KERNEL USAGE STATISTICS")
    print("="*60)
    if CALL_STATS:
        for func_name, count in sorted(CALL_STATS.items(), key=lambda x: x[1], reverse=True):
            print(f"  {func_name}: {count} calls")
    else:
        print("  No custom kernels were called.")
    print("="*60 + "\n")


def main():

    total_time = 0
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

        # give warmup set
        if i == 0:
            continue
        total_time += end - start
    print(total_time)
    
if __name__ == "__main__":
    main()