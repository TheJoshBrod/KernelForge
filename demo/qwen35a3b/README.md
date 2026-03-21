# Qwen 3.5 35B A3B Demo Assets

These files are meant for a live Kernel Forge GUI demo with the official
checkpoint `Qwen/Qwen3.5-35B-A3B`.

## What to upload in the GUI

1. Upload `model.py`.
2. Upload `validation.zip` built from `prompts.jsonl`.
3. Do **not** upload weights unless you have already converted the checkpoint to
   a single `.pt` / `.pth` file.

The wrapper is designed to load from:

- `/home/gb10/model-cache/Qwen3.5-35B-A3B` if it exists
- otherwise `Qwen/Qwen3.5-35B-A3B` from Hugging Face

## Important repo fit notes

- The official model is distributed as sharded `safetensors`, not a single
  uploadable `.pt` / `.pth`.
- This repo requires `model.py`; weights alone are not enough for project
  creation.
- The wrapper forces `attn_implementation="eager"` so the profiler has a chance
  to capture standard PyTorch functional ops.

## Recommended demo flow

1. Pre-download the model before the talk:

   ```bash
   /home/gb10/Projects/Kernal-Forge/CGinS/.venv/bin/python \
     /home/gb10/Projects/Kernal-Forge/CGinS/demo/qwen35a3b/download_model.py
   ```

2. Start the GUI.
3. Create a new project with:
   - backend: `triton` first
   - model file: `demo/qwen35a3b/model.py`
   - validation zip: `demo/qwen35a3b/validation.zip`
   - weights: leave empty
4. Let profiling finish.
5. Inspect captured operators and target the hot ones first.

## Suggested operators for a live run

Start with one operator at a time:

- `torch_nn_functional_linear`
- `torch_nn_functional_softmax`
- `torch_nn_functional_silu`
- `torch_nn_functional_pad`
- `torch_nn_functional_conv1d`

The exact captured set depends on the forward path exercised by the prompts.

## Presentation advice

- For the live demo, show profile -> generate -> optimize on a single operator.
- For the slide deck, precompute multi-operator results and show the saved
  benchmark artifacts instead of waiting live.
