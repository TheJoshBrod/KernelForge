import json
import math
from pathlib import Path
from typing import Dict, List, Optional

from src.optimizer.core.types import KernelNode
from src.optimizer.config.settings import settings

_NODE_CACHE: Dict[int, KernelNode] = {}

def load_tree_once(paths: dict):
    """Populates the cache once at startup."""
    global _NODE_CACHE
    node_path = paths["proj_dir"] / "nodes"
    _NODE_CACHE.clear()
    
    # Check if dir exists first
    if not node_path.exists():
        return

    for file in node_path.glob("*.json"):
        try:
            with open(file, "r") as f:
                data = json.load(f)
                node = KernelNode.model_validate(data)
                _NODE_CACHE[node.id] = node
        except Exception as e:
            print(f"Skipping corrupted node {file}: {e}")

def choose_optimization(paths: dict, C: float = settings.mcts_c_constant, exclude_ids: set = None) -> KernelNode:
    """Select the most promising node to optimize next.
    
    Args:
        paths: Dictionary containing project paths
        C: UCT exploration constant
        exclude_ids: Set of node IDs to skip (for parallel processing)
    
    Returns:
        The selected KernelNode to optimize
    """
    if exclude_ids is None:
        exclude_ids = set()

    # 1. Ensure Cache is Populated
    if not _NODE_CACHE:
        load_tree_once(paths)
        if not _NODE_CACHE:
             raise ValueError("No nodes found. Initialize the tree with a root first.")

    # 2. Identify Roots (from memory)
    roots = [n for n in _NODE_CACHE.values() if n.parent_id == -1]
    if not roots:
        raise ValueError("No root node found")

    # 3. Find most promising leaf
    best_leaf = None
    best_leaf_score = float("inf")

    for root in roots:
        current = root
        
        # Limit depth to prevent infinite loops in weird graph cycles
        depth_sanity = 0
        
        while current.children_ids and depth_sanity < 1000:
            depth_sanity += 1

            # Linear Annealing: From 0.5 down to 0.3            
            total_steps_done = roots[0].visits
            max_steps = 1000

            progress = min(1.0, total_steps_done / max_steps)
            current_exponent = 0.5 - (0.2 * progress)
            max_children = math.floor(current.visits ** current_exponent)
            
            # Branching Condition
            if len(current.children_ids) <= max_children:
                # Return current node so the caller creates a NEW child
                # Skip if this specific node is being processed
                if current.id not in exclude_ids:
                    return current
                # Node is excluded (in-flight), but don't give up on this subtree!
                # Fall through to UCT selection below to find a valid child

            # --- UCT Selection ---
            best_child_node = None
            best_score = float("inf")
            valid_children_found = False

            for child_id in current.children_ids:
                if child_id not in _NODE_CACHE:
                    continue  # Skip missing nodes
                if child_id in exclude_ids:
                    continue  # Skip nodes being processed by other workers
                
                child = _NODE_CACHE[child_id]
                
                # CRITICAL: If a child is 'failed' (inf value), do we visit it?
                # In minimization, 'inf' score is bad, so UCT naturally avoids it.
                # But we must ensure uct_score handles 'inf' math correctly.
                try:
                    score = child.uct_score(current, C)
                except Exception:
                    score = float('inf')

                if score < best_score:
                    best_score = score
                    best_child_node = child
                    valid_children_found = True

            # If all children are broken/failed, or no valid children found
            if not valid_children_found or best_child_node is None:
                # Force progressive widening to escape this dead end
                return current

            current = best_child_node

        # Evaluate Leaf
        # Prefer subtree value, fallback to own value
        val = current.best_subtree_value if current.best_subtree_value is not None else current.value
        
        # Handle case where value might be None (compilation error leaf)
        if val is None: val = float('inf')
        
        # Skip if this node is already being processed
        if current.id in exclude_ids:
            continue

        if val < best_leaf_score:
            best_leaf_score = val
            best_leaf = current

    return best_leaf if best_leaf else roots[0]


def select_n_distinct(paths: dict, n: int, in_flight_ids: set = None) -> list:
    """Select N distinct nodes for parallel processing.
    
    Args:
        paths: Dictionary containing project paths
        n: Number of nodes to select
        in_flight_ids: Set of node IDs already being processed
    
    Returns:
        List of KernelNode objects (may be fewer than n if tree is small)
    """
    if in_flight_ids is None:
        in_flight_ids = set()
    
    selected = []
    exclude_ids = in_flight_ids.copy()
    
    for _ in range(n):
        try:
            node = choose_optimization(paths, exclude_ids=exclude_ids)
            if node is None or node.id in exclude_ids:
                break  # No more valid nodes
            selected.append(node)
            exclude_ids.add(node.id)
        except ValueError:
            break  # No nodes available
    
    return selected

def update_tree(paths: dict, new_node: KernelNode):
    global _NODE_CACHE
    
    base_path = paths["proj_dir"] / "nodes"
    
    # 1. Update Cache Immediately
    _NODE_CACHE[new_node.id] = new_node
    
    current = new_node
    # Normalize value for math
    current_best_val = current.best_subtree_value if current.best_subtree_value is not None else current.value
    if current_best_val is None: current_best_val = float('inf')

    while True:
        # Write CURRENT node to disk (Persist the changes from previous loop iteration)
        node_file = base_path / f"{current.id}.json"
        with open(node_file, 'w') as f:
             f.write(current.model_dump_json(indent=4, by_alias=True))

        if current.parent_id == -1 or current.parent_id is None:
            break

        # Load Parent (From Cache else from disk)
        parent_id = current.parent_id
        if parent_id in _NODE_CACHE:
            parent = _NODE_CACHE[parent_id]
        else:
            parent_file = base_path / f"{parent_id}.json"
            if not parent_file.exists(): break
            with open(parent_file, 'r') as f:
                parent = KernelNode.model_validate(json.load(f))
                _NODE_CACHE[parent_id] = parent

        # --- Update Logic ---
        
        # 1. Link Child
        if current.id not in parent.children_ids:
            parent.children_ids.append(current.id)

        # 2. Visits
        parent.visits += 1

        # 3. Propagate Best Score (Minimization)
        parent_best = parent.best_subtree_value if parent.best_subtree_value is not None else parent.value
        if parent_best is None: parent_best = float('inf')
        
        # 4. Update parent's best score if current is better
        if current_best_val < parent_best:
            parent.best_subtree_value = current_best_val
        else:
            current_best_val = parent_best

        # Move Up
        current = parent

def collect_ancestry(paths: dict, start_node: KernelNode, code_depth: int = 1) -> tuple[list[dict], list[tuple[int, str]]]:
    """
    Walks up the tree from start_node to the root to collect:
    1. The improvement log (metadata for all nodes in the path).
    2. The source code (only for the last 'code_depth' nodes).
    
    Returns:
        (improvement_log, ancestor_codes)
        - improvement_log is sorted Chronologically (Root -> Leaf)
        - ancestor_codes is sorted Chronologically (Oldest -> Newest)
    """
    history = []
    codes = []
    current = start_node
    
    # 1. Traverse Bottom-Up (Child -> Parent -> Root)
    while current:
        # --- Collect Metadata ---
        # Ensure we have a valid runtime for metrics
        runtime = current.value if current.value is not None else float('inf')
        
        entry = {
            "iteration": current.id,
            "attempted": current.improvement_description or "Baseline",
            "results": {
                "mean_time_ms": runtime
            },
            "speedup_vs_parent": getattr(current, "speedup_vs_parent", 1.0) or 1.0
        }
        history.append(entry)

        # --- Collect Code (Sliding Window) ---
        if len(codes) < code_depth:
            try:
                code_path = Path(current.code)
                # Handle relative paths if necessary, though absolute is safer
                if not code_path.is_absolute():
                     code_path = paths["proj_dir"].parent / code_path

                if code_path.exists():
                    codes.append((current.id, code_path.read_text()))
                else:
                    # Fallback if file is missing (shouldn't happen)
                    codes.append((current.id, "// Error: Code file not found on disk."))
            except Exception as e:
                codes.append((current.id, f"// Error reading code: {e}"))

        # --- Move to Parent ---
        if current.parent_id == -1 or current.parent_id is None:
            break

        # Try to load parent from memory cache first
        if current.parent_id in _NODE_CACHE:
            current = _NODE_CACHE[current.parent_id]
        else:
            p_path = paths["proj_dir"] / "nodes" / f"{current.parent_id}.json"
            if p_path.exists():
                try:
                    with open(p_path, 'r') as f:
                        current = KernelNode.model_validate(json.load(f))
                except Exception as e:
                    raise RuntimeError(f"CRITICAL: Corrupted parent node file at {p_path}") from e
            else:
                raise FileNotFoundError(f"CRITICAL: Orphaned node detected. Parent file {p_path} is missing for child {current.id}")

    # 2. Reorder (Root -> Child)
    history.reverse()
    codes.reverse()

    # 3. Calculate "Speedup vs Baseline"
    if history:
        baseline_time = history[0]["results"]["mean_time_ms"]
        
        for h in history:
            current_time = h["results"]["mean_time_ms"]
            if current_time > 0 and baseline_time > 0:
                h["speedup_vs_baseline"] = baseline_time / current_time
            else:
                h["speedup_vs_baseline"] = 0.0

    return history, codes


def get_existing_roots(paths: dict) -> list[dict]:
    """Get all root nodes with their kernel code for diversity guidance.
    
    Used when creating new roots to show the LLM what approaches already exist.
    
    Args:
        paths: Dictionary containing project paths
        
    Returns:
        List of dicts with {id, runtime_ms, code_preview} sorted by runtime (best first)
    """
    node_path: Path = paths["proj_dir"] / "nodes"
    roots = []
    
    for file in node_path.glob("*.json"):
        try:
            with open(file, 'r') as f:
                node = json.load(f)
            
            # Only include root nodes (parent == -1)
            if node.get("parent") != -1:
                continue
            
            # Read the kernel code - handle both absolute and relative paths
            code_path_str = node.get("code", "")
            code_path = Path(code_path_str)
            
            # If relative path, resolve from project parent directory
            if not code_path.is_absolute():
                # e.g., "torch_nn_functional_embedding/attempts/kernel_0.cu"
                # proj_dir is like ".../NVIDIA.../torch_nn_functional_embedding"
                # so parent is ".../NVIDIA..." and we join with relative path
                code_path = paths["proj_dir"].parent / code_path_str
            
            if code_path.exists():
                code_preview = code_path.read_text()[:4000]  # First 4KB
            else:
                code_preview = f"// Code file not found: {code_path}"
            
            roots.append({
                "id": node["id"],
                "runtime_ms": node.get("value", 0.0),
                "code_preview": code_preview
            })
        except Exception as e:
            print(f"Warning: Error loading root node from {file}: {e}")
            continue
    
    # Sort by runtime (best/fastest first)
    return sorted(roots, key=lambda x: x["runtime_ms"])