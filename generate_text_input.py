import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

df = pd.DataFrame({
    "sentence": [
        "The quick brown fox jumps over the lazy dog.",
        "AI is transforming the world.",
        "CUDA kernels can accelerate deep learning.",
        "Transformers are powerful models.",
        "PyTorch is popular for research.",
        "Distributed training improves scale.",
        "Profiling kernels identifies bottlenecks.",
        "High bandwidth memory helps GPUs.",
        "Kernel fusion improves efficiency.",
        "Optimizers help convergence.",
        "Benchmarks measure performance.",
        "HuggingFace provides great models.",
        "Custom kernels replace PyTorch ops.",
        "GPU utilization matters.",
        "Latency is critical.",
        "Stable Diffusion uses U-Net.",
        "Multi-head attention is expensive.",
        "Embedding tables can be huge.",
        "Dropout is stochastic regularization.",
        "Linear layers dominate compute."
    ]
})

table = pa.Table.from_pandas(df)
pq.write_table(table, "benchmarks/generate_benchmarks/text_inputs.parquet")

print("Created benchmarks/generate_benchmarks/text_inputs.parquet")
