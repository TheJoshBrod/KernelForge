<div align="center">

# <img src="frontend/assets/KernelForge.svg" alt="KernelForge" width="200"/> <br> Kernel Forge

**Drop-in GPU kernel optimizer for PyTorch models.**

![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Triton](https://img.shields.io/badge/Triton-000000?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAOCAMAAAAsYw3eAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAzFBMVEUAAAAsuMU2vMczu8Ynt8MitcJOxM4pt8QktsIgtMEruMQltsMhtcIjtcIgtMEltsMhtcI6vclozNVpzdU7vckitcIwusZRxc9IwsxGwcxHwcxSxc8ftME0u8dYx9FCwMtMw81ayNFBv8s1u8cetME/v8pEwMty0NhVxtBTxc9Av8o+vspcyNJKws1UxtBLw81byNI9vslPxM5eydJOxM5nzNVmzNU4vMhz0NhFwctqzdY5vch10dkuucVMw85Nw84vucVWxtA8vsn////g7i7RAAAAD3RSTlMAAAAIVdYBLJzxGXbgyvw6EdVsAAAAAWJLR0RDZ9ANYgAAAAd0SU1FB+oDEhMzJWe0294AAACYSURBVAjXVY5HEoJAFAU/OYMKkhyRUcKACiIq5nT/Qzmy0mXXq+p+AMBwvCDwHAMArCjJimUpsiSyoGr6YDiynbGuqWCYrucHoT9xTQPQNJrFGMfzaIEgSbOcFAXJszKBZBmu1hWpN34Pzbat63bXpBRKXO2D4FBhuqDu2J3Odna5doiqb949DB/Fk6q/UYe83n30787v0Q+e/xC9ZxRPWgAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyNi0wMy0xOFQxOTo1MTozNyswMDowMOap+1gAAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjYtMDMtMThUMTk6NTE6MzcrMDA6MDCX9EPkAAAAAElFTkSuQmCC)
![More coming soon](https://img.shields.io/badge/more_coming_soon...-555555?style=for-the-badge)

[![Discord](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/cchEQguDB7)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---
</div>

Kernel Forge automatically generates and optimizes GPU kernels for PyTorch models with no kernel programming expertise required. It profiles your model at the operator level, uses an LLM to write a correct kernel, then searches for performance improvements using Monte Carlo Tree Search until the kernel beats PyTorch's baseline.

---

## Who is this for?

- **ML engineers running models in production** who want lower inference latency on specific hardware without writing CUDA or Triton by hand.
- **AI infrastructure teams** targeting specific GPU hardware (NVIDIA CUDA or AMD ROCm) who need kernels tuned to that exact device.
- **Teams with remote GPU access** who run optimization on a separate GPU server while managing projects locally.
- **Researchers** benchmarking operator-level speedups across different LLM backends or optimization strategies.
- **Teams packaging models for deployment** who want a self-contained inference artifact with kernels baked in and no runtime dependency on KernelForge.

---

## Features

- Automated kernel generation via LLM with compile-error feedback loop
- MCTS-driven optimization - explores tiling, loop unrolling, vectorized memory access, and more
- CUDA and Triton backends (NVIDIA and AMD ROCm)
- Remote execution over SSH - no local GPU required
- Multi-LLM support: Anthropic, OpenAI, Google
- Web dashboard with live progress, speed charts, and MCTS tree inspector
- Portable `.anvil` snapshots and self-contained `.cast` inference packages

[Full feature details](docs/features.md)

---

## Benchmark Snapshot

### Qwen 3.5 35B-A3B

On this mixed-workload run, `Kernel Forge mixed latest` delivered the best overall result against both PyTorch eager and `torch.compile`.

- Total latency: `3693.6 ms` vs `4193.3 ms` for PyTorch eager and `4546.5 ms` for `torch.compile`
- Relative to eager throughput: `1.09x` prefill tok/s, `1.14x` decode tok/s, and `1.13x` total tok/s
- In this run, `torch.compile` slightly improved prefill (`1.02x`) but regressed decode (`0.92x`) and total throughput (`0.92x`) relative to eager

<table>
<tr>
<td><img src="docs/benchmarks/qwen35_mixed_latency_breakdown.png" alt="Qwen 3.5 35B-A3B latency breakdown"/></td>
<td><img src="docs/benchmarks/qwen35_mixed_throughput_vs_eager.png" alt="Qwen 3.5 35B-A3B throughput vs PyTorch eager"/></td>
</tr>
</table>

---

## Quick start

See [system requirements](docs/requirements.md) before installing.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd frontend
jac install
```

Configure your LLM key in the settings panel after starting, or set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_API_KEY` before launch.

```bash
jac start main.jac
```

Open `http://localhost:8000`. Create a project, upload your model weights, and click **Start Forge**.

---

## CLI

For headless or scripted runs, see [docs/cli.md](docs/cli.md).

---

## Further reading

- [System requirements](docs/requirements.md)
- [CLI reference](docs/cli.md)
- [System architecture](docs/system-architecture.md)
- [File formats (.anvil, .cast)](docs/FileFormat.md)
- [Cast runtime](docs/cast-runtime.md)
- [Profiling API](docs/profiling/api.md)
- [Profiling architecture](docs/profiling/architecture.md)
- [Backend source layout](src/README.md)
- [Frontend walker API](frontend/README.md)
