# System Requirements

## Local machine

| Requirement | Notes |
|-------------|-------|
| Python 3.12+ | Required |
| PyTorch | Any build; CUDA build needed for local CUDA/Triton work |
| `jac` | Required to run the web frontend |
| LLM API key | Anthropic, OpenAI, or Google |

A GPU is not required on the local machine if you use remote execution over SSH.

## For local CUDA kernel generation

| Requirement | Notes |
|-------------|-------|
| NVIDIA GPU | Any CUDA-capable device |
| CUDA Toolkit 11.8+ | Required for JIT kernel compilation |
| NVIDIA driver >= 525 | Required for `nvidia-ml-py` profiling |
| `ninja` | Faster JIT compilation (installed via `requirements.txt`) |

## For local Triton kernel generation

| Requirement | Notes |
|-------------|-------|
| NVIDIA GPU or AMD ROCm GPU | Both supported |
| Triton | Installed via `requirements.txt` |
| `rocm-smi` | Required for AMD GPU device spec detection |

## Remote execution (SSH)

The remote host needs Python 3 and a CUDA or ROCm GPU. KernelForge installs everything else automatically on first connect.
