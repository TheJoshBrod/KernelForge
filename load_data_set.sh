#!/usr/bin/env bash
set -e

# Sets up initial raw dataset (imgs, txt, etc.)
./benchmarks/data/request_images.sh
./benchmarks/data/request_text.sh

# Runs the model profiler on dataset to track input outputs of each PyTorch Op

read -r -p "Delete and Reset benchmarks/profiler/individual_ops (useful for new raw-dataset or models)? [y/N]: " confirm
if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
    rm -rf benchmarks/profiler/individual_ops
    python3 benchmarks/profiler/benchmark_generator.py
else
  echo "Skipping deletion."
fi
