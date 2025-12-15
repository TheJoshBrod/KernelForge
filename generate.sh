#!/usr/bin/env bash
set -e

PYTHON=python3

$PYTHON - <<'EOF'
import sys
import subprocess

print("Checking for Required Python Libraries")

required_modules = {
    # ML / data
    "torch": "torch",
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",

    # gRPC / protobuf
    "protobuf": "google.protobuf",
    "grpcio": "grpc",
    "grpcio-status": "grpc_status",

    # AI / LLM clients
    "google-generativeai": "google.generativeai",
    "openai": "openai",
    "ollama": "ollama",
    "anthropic": "anthropic",

    # Tooling
    "ninja": "ninja",
    "python-dotenv": "dotenv",
}

missing = []

for pkg, module in required_modules.items():
    try:
        __import__(module)
    except Exception:
        missing.append(pkg)

if missing:
    print("Missing required Python libraries:", file=sys.stderr)
    for pkg in missing:
        print(f"  - {pkg}", file=sys.stderr)
    print("Installing libraries...")
    for pkg in missing:
        print(f"\t- {pkg}", file=sys.stderr)
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


print("All required Python libraries are installed.")
EOF

python -m src.generator.main benchmarks/data/PyTorchFunctions/
