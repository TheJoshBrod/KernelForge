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

def choose_optimization(paths: dict, C: float = settings.mcts_c_constant) -> KernelNode:

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
                return current

            # --- UCT Selection ---
            best_child_node = None
            best_score = float("inf")
            valid_children_found = False

            for child_id in current.children_ids:
                if child_id not in _NODE_CACHE:
                    continue # Skip missing nodes
                
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

        if val < best_leaf_score:
            best_leaf_score = val
            best_leaf = current

    return best_leaf if best_leaf else roots[0]

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