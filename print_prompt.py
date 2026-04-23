import json
import os

proj_dir = "/home/jodab/KernelForge/kernels/projects/resnet - siblings - GTX 1660 Ti"
with open(os.path.join(proj_dir, "io", "dag.json"), "r") as f:
    dag_data = json.load(f)
dag_nodes = dag_data.get("nodes", [])
dag_edges = dag_data.get("edges", [])

bench_map = {}
bench_path = os.path.join(proj_dir, "benchmarks", "op_benchmarks.json")
if os.path.exists(bench_path):
    with open(bench_path, "r") as f:
        bench_data = json.load(f)
    for r in bench_data.get("results", []):
        bench_op = r.get("op", "")
        if bench_op:
            short = bench_op[len("torch_nn_functional_"):] if bench_op.startswith("torch_nn_functional_") else bench_op
            bench_map[short] = r

node_ops = {n.get("id"): n.get("op") for n in dag_nodes if n.get("id") and n.get("op")}

adj_out = {}
adj_in = {}
for edge in dag_edges:
    src = edge.get("source")
    tgt = edge.get("target")
    if src and tgt:
        adj_out.setdefault(src, []).append(tgt)
        adj_in.setdefault(tgt, []).append(src)

visited = set()
all_chains = []
for n in dag_nodes:
    nid = n.get("id")
    if not nid or nid in visited: continue
    
    head = nid
    backward_visited = {nid}
    while True:
        preds = adj_in.get(head, [])
        if len(preds) != 1: break
        pred = preds[0]
        if len(adj_out.get(pred, [])) != 1: break
        if pred in visited or pred in backward_visited: break
        backward_visited.add(pred)
        head = pred
        
    chain = [head]
    visited.add(head)
    current = head
    while True:
        outs = adj_out.get(current, [])
        if len(outs) != 1: break
        next_id = outs[0]
        if len(adj_in.get(next_id, [])) != 1: break
        if next_id in visited: break
        chain.append(next_id)
        visited.add(next_id)
        current = next_id
        
    if len(chain) >= 2:
        all_chains.append(chain)

pattern_groups = {}
for chain in all_chains:
    if len(chain) < 2: continue
    op_seq_list = [node_ops.get(nid, "?") for nid in chain]
    key = "|".join(op_seq_list)
    if key not in pattern_groups:
        pattern_groups[key] = {"ops": op_seq_list, "instances": []}
    pattern_groups[key]["instances"].append(chain)

sorted_patterns = sorted(pattern_groups.items(), key=lambda x: -len(x[1]["instances"]))
pattern_lines = []
for key, pg in sorted_patterns[:20]:
    ops = pg["ops"]
    freq = len(pg["instances"])
    combined_ms = 0.0
    for op in ops:
        pm = bench_map.get(op, {}).get("pytorch_ms")
        if pm:
            try: combined_ms += float(pm)
            except: pass
    ms_str = f"{combined_ms:.3f}ms" if combined_ms > 0 else "N/A"
    pattern_lines.append(f"  [{', '.join(ops)}] x{freq} — {ms_str}/instance")

op_timing_lines = []
for short_op in list(sorted(bench_map.keys()))[:30]:
    r = bench_map[short_op]
    pm = r.get("pytorch_ms")
    km = r.get("kernel_ms")
    winner = r.get("winner", "")
    if pm is not None:
        try:
            pm = float(pm)
            status_tag = " [FORGED]" if winner == "optimized" else ""
            km_str = f" / kernel={float(km):.3f}ms" if km is not None else ""
            op_timing_lines.append(f"  {short_op}: pytorch={pm:.3f}ms{km_str}{status_tag}")
        except Exception as e:
            pass

pattern_text = "\n".join(pattern_lines) if pattern_lines else "  (no repeated patterns)"
op_timing_text = "\n".join(op_timing_lines) if op_timing_lines else "  (no benchmark data)"
total_nodes = len(dag_nodes)

system_prompt = """You are a GPU kernel fusion expert analyzing a neural network computation graph.
Identify which adjacent operator chains are strong candidates for kernel fusion to reduce global memory traffic and kernel launch overhead.

Return ONLY valid JSON (no markdown fences, no text outside JSON) in this exact schema:
{
  "fusion_groups": [
    {
      "name": "short_identifier",
      "pattern_ops": ["op1", "op2", "op3"],
      "estimated_speedup": 1.8,
      "rationale": "one sentence explanation"
    }
  ]
}

Rules:
- Only propose fusions for ops in the listed patterns (adjacent in the graph)
- Prioritize high-frequency patterns (many instances) for maximum model-wide impact
- Skip patterns where all ops are already [FORGED]
- Common wins: conv+batch_norm+relu, linear+gelu, linear+relu, adaptive_avg_pool2d+linear
- estimated_speedup must be between 1.1 and 3.5"""

user_prompt = f"""Project: resnet - siblings - GTX 1660 Ti
Total ops in graph: {total_nodes}

Repeated adjacent-op patterns (op sequence → instances → combined timing/instance):
{pattern_text}

Per-op baseline timing:
{op_timing_text}

Propose the best fusion candidates."""

print("=== SYSTEM PROMPT ===")
print(system_prompt)
print("\n=== USER PROMPT ===")
print(user_prompt)
