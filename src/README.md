# src

This directory contains the main code for the generator and optimizer.

```
src/
├── README.md
├── generator/
│   ├── generator.py
│   ├── improvement.py
│   ├── logger.py
│   ├── main.py
│   ├── monitor.py
│   ├── prompts
│   │   ├── GeneratorSystemPrompt.md
│   │   ├── __pycache__
│   │   │   └── prompts.cpython-312.pyc
│   │   └── prompts.py
│   └── verifier.py
└── optimizer/
    ├── generate_all_optimized_kernels.py
    ├── generate_text_input.py
    └── optimize_kernel.py
```

## Generator

This directory contains the pipeline for generating initial batches of validate kernels. These kernels are only tested for correctness, NOT PERFORMANCE. 

The purpose of this pipeline is to generate your initial batch of correct kernel to allow for few-shot prompting later in the optimizer stage. 

To run the pipeline run the `generate.sh` file in the top level directory with appropriate flags

## Optimizer

This directory contains the pipeline for generating secondary batches of kernels that are optimized for your specific hardware. These kernels can be configured to optimize for varying heuristics: inference time, GPU utilization, etc. 

The purpose of this pipeline is to improve the initial batch of correct kernel using the validated kernels previously generated for few-shot prompting. 

To run the pipeline run the `optimize.sh` file in the top level directory with appropriate flags

