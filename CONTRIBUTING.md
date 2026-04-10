# Contributing to KernelForge

Thanks for your interest in contributing. This document covers how to get set up, what areas need work, and how to submit changes.

---

## Table of contents

- [Getting started](#getting-started)
- [Project layout](#project-layout)
- [Running tests](#running-tests)
- [Areas to contribute](#areas-to-contribute)
- [Submitting changes](#submitting-changes)
- [Code style](#code-style)

---

## Getting started

**Prerequisites:** Python 3.12+, a CUDA-capable GPU (or SSH access to one), and an LLM API key (Anthropic, OpenAI, or Google).

```bash
git clone https://github.com/TheJoshBrod/KernelForge.git
cd KernelForge
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd frontend
jac install
jac start main.jac
```

See [docs/requirements.md](docs/requirements.md) for the full list of system dependencies including CUDA Toolkit and driver versions.

---

## Project layout

```
src/
  generator/       # LLM-driven kernel generation (correctness pipeline)
  optimizer/
    core/          # MCTS search logic
    backends/
      cuda/        # CUDA JIT compilation, verification, profiling
      triton/      # Triton backend (NVIDIA + AMD ROCm)
  llm/             # LLM client abstractions (Anthropic, OpenAI, Google)
  projects/        # Project state and kernel node tree persistence
frontend/          # Web dashboard (Jaclang)
kernels/           # Saved kernel outputs
tests/             # Test suite
docs/              # Documentation
```

Key entry points:
- `src/generator/generator.py` - LLM kernel generation loop
- `src/optimizer/core/mcts.py` - MCTS selection and tree update
- `src/optimizer/backends/cuda/verifier.py` - correctness verification
- `src/optimizer/backends/cuda/profiler.py` - benchmarking

---

## Running tests

```bash
pytest tests/
```

The test suite includes benchmark harness tests. If your change touches the profiling or verification pipeline, make sure these pass before submitting. Additionally before adding a new feature, you must create a new test to the harness.

---

## Areas to contribute

- **New optimization strategies** — MCTS currently explores tiling, loop unrolling, and vectorized memory access. New strategies can be added as prompts in the optimizer.
- **Backend support** — Triton support is in progress; Metal or other backends would be welcome.
- **LLM providers** — New provider integrations go in `src/llm/`.
- **Frontend improvements** — The web dashboard is written in Jac. See `frontend/README.md` for the walker API.
- **Documentation** — Anything missing or unclear in `docs/`.
- **Bug reports** — Open an issue with your GPU model, CUDA version, Python version, and the full traceback.

---

## Submitting changes

1. Fork the repo and create a branch from `main`.
2. Make your changes with focused, self-contained commits.
3. Run the test suite and confirm it passes.
4. Open a pull request against `main` with a clear description of what the change does and why.

For larger changes (new backends, architectural changes), open an issue first to discuss the approach before writing code.

---

## Code style

- Python: follow PEP 8, no lines over 120 characters.
- No unnecessary abstractions, if something is used once, keep it inline. If unsure attach it to the issue.
- Don't add docstrings or comments to code you didn't write.

---

Questions? Join the [Discord](https://discord.gg/cchEQguDB7) or open an issue!
