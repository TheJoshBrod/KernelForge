import torch
import torch.autograd.profiler as profiler
import torch.nn.functional as F
from functools import wraps
from PIL import Image
import os
import inspect
import requests


# from benchmarks.generate_benchmarks.trivial import SimpleModel
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
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
    
    # Get function signature to extract default values
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        sig = None

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = f"{module_path}.{func_name}"

        # Convert args to CPU tensors (lossless)
        ser_args = [
            a.detach().cpu() if isinstance(a, torch.Tensor) else a
            for a in args
        ]

        # Only record kwargs explicitly passed
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

        calls.setdefault(key, []).append({
            "args": ser_args,
            "kwargs": ser_kwargs,
            "output": ser_output
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


# ****************************
#    Initialize PyTorch Model
# ****************************

model_name = "facebook/convnext-tiny-224" 

print(f"Loading {model_name}...")
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

img_path = "test_image.jpg"
if not os.path.exists(img_path):
    print("Downloading test image...")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    image.save(img_path)
else:
    image = Image.open(img_path)

inputs = processor(images=image, return_tensors="pt")

config = AutoConfig.from_pretrained(model_name)
id2label = config.id2label

# ****************************
#        ATen Profiler 
# ****************************

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device_str)
inputs = {k: v.to(device_str) for k, v in inputs.items()}
with profiler.profile(record_shapes=True, use_device=device_str) as prof:
    with profiler.record_function("forward"):
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            top5 = probs.topk(5)
            print("\nRAW TOP-5 TENSOR:\n", top5)

            print("\nDECODED PREDICTIONS:")
            for idx in top5.indices.squeeze().tolist():
                print(f"{idx}: {id2label[idx]} ({float(probs[0][idx]):.5f})")

print(prof.key_averages().table(sort_by="count", row_limit=50))

# ****************************
#  Export PyTorch API In/Out 
# ****************************


base_dir = "benchmarks/generate_benchmarks/PyTorchFunctions"
os.makedirs(base_dir, exist_ok=True)

for func_name, entries in calls.items():
    # Safe filename
    filename = func_name.replace("/", ".") + ".pt"
    file_path = os.path.join(base_dir, filename)

    # If file exists → load and append
    if os.path.exists(file_path):
        existing = torch.load(file_path)
        # Ensure merging (existing must be a dict)
        if not isinstance(existing, dict):
            raise ValueError(f"Expected list in {file_path}, got {type(existing)}")
        
        existing[func_name].extend(entries)

        torch.save(existing, file_path)
        print(f"Updated file: {func_name}")

    else:
        # Create new file
        torch.save(entries, file_path)
        print(f"Created new file: {func_name}")