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

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SKIP_FUNCTIONS = [
    "has_torch_function",
    "handle_torch_function",
    "is_storage",
    "result_type",
    "get_default_dtype",
]

# ****************************
# Track specific PyTorch Calls
# ****************************
calls = {}
_wrapped = set()
ENABLE_WRAPPING = True

def wrap_function(module, func_name):
    if not ENABLE_WRAPPING:
        return
    func = getattr(module, func_name)
    if func in _wrapped:
        return
    _wrapped.add(func)

    module_path = module.__name__
    
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
        key = f"{module_path}.{func_name}"

        # Convert args to CPU tensors (lossless)
        ser_args = [
            a.detach().cpu() if isinstance(a, torch.Tensor) else a
            for a in args
        ]

        # Store kwargs as provided
        ser_kwargs = {
            k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in kwargs.items()
        }

        output = func(*args, **kwargs)

        if isinstance(output, torch.Tensor):
            ser_output = output.detach().cpu()
        elif isinstance(output, (list, tuple)):
            ser_output = [
                o.detach().cpu() if isinstance(o, torch.Tensor) else o
                for o in output
            ]
        else:
            ser_output = output

        calls.setdefault(key, [])
        calls[key].append({
            "args": ser_args,
            "kwargs": ser_kwargs,
            "output": ser_output,
            "signature": {
                "params": params,
                "defaults": defaults
            }
        })

        return output

    setattr(module, func_name, wrapper)

# Wrap all functions in torch.nn.functional
for name in dir(F):
    if name.startswith("_"):
        continue  # skip private names like _in_projection, etc.
    if any(skip in name for skip in ["torch_function", "storage", "result_type", "dtype"]):
        continue  # skip helpers by substring match
    obj = getattr(F, name)
    if callable(obj):
        wrap_function(F, name)


def save_entries(func_name, entries):
    base_dir = "benchmarks/generate_benchmarks/PyTorchFunctions"
    func_dir = os.path.join(base_dir, func_name.replace(".", "_").replace("/", "_"))
    os.makedirs(func_dir, exist_ok=True)
    
    # Count existing files to avoid overwriting
    existing_count = len(glob.glob(os.path.join(func_dir, "entry_*.pt")))
    
    for idx, entry in enumerate(entries):
        if existing_count > 200:
            return
        file_path = os.path.join(func_dir, f"entry_{existing_count + idx:06d}.pt")
        torch.save(entry, file_path)

def profile_image_model(model_name: str = "facebook/convnext-tiny-224"):
    """Runs Image Classification model from HuggingFace's Transformer library to profile input and outputs. 
    Outputs the input/output pairs to pt file separate by PyTorch function name

    Args:
        model_name (str, optional): Name of Hugging Face image classification model. Defaults to "facebook/convnext-tiny-224".
    """
    global calls

    # ****************************
    #    Initialize PyTorch Model
    # ****************************

    print(f"Loading {model_name}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device_str)
 
    # ****************************
    #        Run Model 
    # ****************************

    print(f"Running {model_name}...")
    image_paths = glob.glob("benchmarks/generate_benchmarks/images/*.JPEG")
    for path in tqdm.tqdm(image_paths[:7], desc="images"):
        print(path)
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
                        with torch.no_grad():
                            outputs = model(**inputs)

            # Calculate Output 
            logits = outputs.logits
            predicted_class_id = logits.argmax(-1).item()
            label = id2label[predicted_class_id]
            print(label)
            # ****************************
            #  Export PyTorch API In/Out 
            # ****************************

            print(f"Exporting {model_name}...")
            base_dir = "benchmarks/generate_benchmarks/PyTorchFunctions"
            os.makedirs(base_dir, exist_ok=True)

            for func_name, entries in calls.items():
                save_entries(func_name, entries)
            calls.clear()
            torch.cuda.empty_cache()


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

    global calls

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Load text samples    
    pq_path = "benchmarks/generate_benchmarks/text_inputs.parquet"
    table = pq.read_table(pq_path)
    df = table.to_pandas()
    text_samples = df["sentence"].tolist()[:20]

    print(f"Running {model_name}...")
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device_str)

    for text in tqdm.tqdm(text_samples, desc="Text Samples"):
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

        print(f"TEXT: {text}\nPREDICTION: {label_text}\n")

        # ****************************
        #  Export PyTorch API In/Out
        # ****************************

        for func_name, entries in calls.items():
            save_entries(func_name, entries)
        calls.clear()
        torch.cuda.empty_cache()

    print("Done.")



def main():
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
    
if __name__ == "__main__":
    main()