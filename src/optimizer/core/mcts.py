import json
import math
from pathlib import Path
from typing import Dict, Optional

from src.optimizer.core.types import KernelNode
from src.optimizer.config.settings import settings

def choose_optimization(paths: dict, C: float = settings.mcts_c_constant) -> KernelNode:
    """Chooses which node to branch from on optimization tree

    Args:
        paths (dict): Dictionary containing paths to project files
        C (float): UCT exploration constant

    Returns:
        KernelNode: node to expand
    """
    node_path: Path = paths["proj_dir"] / "nodes"

    # ---- Load all nodes ----
    nodes: Dict[int, KernelNode] = {}

    for file in node_path.glob("*.json"):
        with open(file, "r") as f:
            data = json.load(f)
            node = KernelNode.model_validate(data)
            nodes[node.id] = node

    if not nodes:
        raise ValueError("No nodes found in directory")

    # ---- Identify root(s) ----
    # Root is a node whose parent is -1
    roots = [
        n for n in nodes.values()
        if n.parent_id == -1
    ]

    if not roots:
        raise ValueError("No root node found")

    # Find most promising child of all roots
    best_leaf = None
    best_leaf_score = float("inf")

    for root in roots:
        current = root

        # ---- Traverse using UCT until leaf ----
        while current.children_ids:
            best_child_node = None
            best_score = float("inf")

            for child_id in current.children_ids:
                if child_id not in nodes:
                    continue

                child = nodes[child_id]
                score = child.uct_score(current, C)

                if score < best_score:
                    best_score = score
                    best_child_node = child

            if best_child_node is None:
                # If all children are invalid or something went wrong, treat as leaf
                break

            current = best_child_node

        # ---- Evaluate leaf reached from this root ----
        # Use subtree value if present, else node value
        leaf_score = current.best_subtree_value if current.best_subtree_value is not None else current.value

        if leaf_score < best_leaf_score:
            best_leaf_score = leaf_score
            best_leaf = current

    if best_leaf is None:
        return roots[0]

    # ---- Return node to expand ----
    return best_leaf


def update_tree(paths: dict, new_node: KernelNode):
    """
    Recursively updates the tree moving upwards from the new_node.
    Updates children_ids, visits, and best_subtree_value for ancestors.
    
    Args:
        paths (dict): Dictionary containing project paths (specifically 'proj_dir')
        new_node (KernelNode): The newly added/updated node
    """
    if isinstance(paths, dict):
        base_path = paths["proj_dir"] / "nodes"
    else:
        # Fallback if a Path object is passed directly
        base_path = paths

    current = new_node

    # We need the best value from the current node to propagate up
    current_best_val = current.best_subtree_value if current.best_subtree_value is not None else current.value

    while True:
        # If root, we are done propagating to parents
        if current.parent_id == -1 or current.parent_id is None:
            break

        parent_file = base_path / f"{current.parent_id}.json"
        if not parent_file.exists():
            print(f"Warning: Parent file {parent_file} not found.")
            break

        with open(parent_file, 'r') as f:
            data = json.load(f)
            parent = KernelNode.model_validate(data)

        # 1. Update children list
        if current.id not in parent.children_ids:
            parent.children_ids.append(current.id)

        # 2. Only update the immediate parent's visits, not all ancestors
        if current.id == new_node.id:
             parent.visits += 1

        # 3. Update best_subtree_value
        parent_best = parent.best_subtree_value if parent.best_subtree_value is not None else parent.value
        
        # If the child's best value is better than parent's current best, update parent
        if current_best_val is not None and (parent_best is None or current_best_val < parent_best):
            parent.best_subtree_value = current_best_val
            # If we updated the best, this new best is what propagates further up
            current_best_val = parent.best_subtree_value
        else:
            # If the child didn't improve the parent, propagate the parent's existing best
            current_best_val = parent_best

        # Save updates
        with open(parent_file, 'w') as f:
            f.write(parent.model_dump_json(indent=4, by_alias=True))

        # Move up
        current = parent


def collect_ancestry(paths: dict, leaf_node: KernelNode, code_depth: int = 3) -> tuple[list[dict], list[tuple[int, str]]]:
    """
    Traverse from leaf to root, collecting:
    1. All improvement descriptions (full path to root)
    2. Last N kernel code snippets with their iteration IDs
    
    Args:
        paths: Project paths dict
        leaf_node: Node to start from
        code_depth: How many ancestor kernel codes to include
        
    Returns:
        (improvement_log, ancestor_codes)
        - improvement_log: List of {iteration, attempted, results, speedup_vs_baseline}
        - ancestor_codes: List of (iteration_id, code_string) tuples (most recent first, then reversed)
    """
    node_path: Path = paths["proj_dir"] / "nodes"
    
    # Load baseline (node 0) to calculate true speedup vs baseline
    baseline_value = None
    baseline_file = node_path / "0.json"
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            baseline_node = KernelNode.model_validate(json.load(f))
            baseline_value = baseline_node.value  # baseline runtime in ms
    
    improvement_log = []
    ancestor_codes = []  # Now stores (iteration_id, code) tuples
    current = leaf_node
    
    while current is not None:
        # Collect improvement description for ALL ancestors (except baseline)
        if current.improvement_description and current.improvement_description != "Initial":
            # Calculate TRUE speedup vs baseline (baseline_time / current_time)
            if baseline_value and current.value and current.value > 0:
                speedup_vs_baseline = baseline_value / current.value
            else:
                speedup_vs_baseline = 1.0
            
            improvement_log.append({
                "iteration": current.id,
                "attempted": current.improvement_description,
                "results": {"mean_time_ms": current.value},
                "speedup_vs_baseline": speedup_vs_baseline,
                "speedup_vs_parent": current.speedup_vs_parent or 1.0
            })
        
        # Collect kernel code for only N ancestors (with iteration ID)
        if len(ancestor_codes) < code_depth and current.code:
            code_path = Path(current.code)
            if code_path.exists():
                ancestor_codes.append((current.id, code_path.read_text()))
        
        # Move to parent
        if current.parent_id == -1 or current.parent_id is None:
            break
            
        parent_file = node_path / f"{current.parent_id}.json"
        if parent_file.exists():
            with open(parent_file, 'r') as f:
                current = KernelNode.model_validate(json.load(f))
        else:
            break
    
    # Reverse both so oldest is first (matches iteration order)
    improvement_log.reverse()
    ancestor_codes.reverse()
    
    return improvement_log, ancestor_codes

