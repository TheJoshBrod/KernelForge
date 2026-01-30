import json
import math
from pathlib import Path
from typing import Dict

class kernel_node:
    def __init__(self, data: dict):
        self.id = data["id"]
        self.code = data.get("code")
        self.value = data.get("value")
        self.parent_id = data.get("parent")
        self.children_ids = data.get("children", [])
        self.visits = data.get("visits", 1)
        self.best_subtree_value = data.get("best_subtree_value")
        self.data = data

    def uct_score(self, parent_node: 'kernel_node', C: float = 1.0) -> float:
        """
        Compute UCT score for minimization.
        Lower score = better.
        """

        exploitation = self.best_subtree_value if self.best_subtree_value is not None else self.value
        exploration = C * math.sqrt(math.log(parent_node.visits) / self.visits)

        return exploitation - exploration

def choose_optimization(node_path: Path, C: float = 1.0) -> kernel_node:
    """Chooses which node to branch from on optimization tree

    Args:
        node_path (Path): Directory containing node JSON files
        C (float): UCT exploration constant

    Returns:
        kernel_node: node to expand
    """

    # ---- Load all nodes ----
    nodes: Dict[int, kernel_node] = {}

    for file in node_path.glob("*.json"):
        with open(file, "r") as f:
            data = json.load(f)
            node = kernel_node(data)
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


def update_tree(node_path: Path, new_node: kernel_node):
    """
    Recursively updates the tree moving upwards from the new_node.
    Updates children_ids, visits, and best_subtree_value for ancestors.
    
    Args:
        node_path (Path): Directory containing node JSON files
        new_node (kernel_node): The newly added/updated node
    """
    current = new_node
    
    # We need the best value from the current node to propagate up
    current_best_val = current.best_subtree_value if current.best_subtree_value is not None else current.value

    while True:
        # If root, we are done propagating to parents
        if current.parent_id == -1:
            break

        parent_file = node_path / f"{current.parent_id}.json"
        if not parent_file.exists():
            print(f"Warning: Parent file {parent_file} not found.")
            break
            
        with open(parent_file, 'r') as f:
            data = json.load(f)
            parent = kernel_node(data)
            
        # 1. Update children list
        if current.id not in parent.children_ids:
            parent.children_ids.append(current.id)
            parent.data['children'] = parent.children_ids
            
        # 2. Update visits
        parent.visits += 1
        parent.data['visits'] = parent.visits
        
        # 3. Update best_subtree_value
        parent_best = parent.best_subtree_value if parent.best_subtree_value is not None else parent.value
        
        if current_best_val < parent_best:
            parent.best_subtree_value = current_best_val
            parent.data['best_subtree_value'] = current_best_val
            # If we updated the best, this new best is what propagates further up
            current_best_val = parent.best_subtree_value
        else:
            # If the child didn't improve the parent, the parent's existing best
            # is what propagates up (which might be from another branch)
            current_best_val = parent_best

        # Save updates
        with open(parent_file, 'w') as f:
            json.dump(parent.data, f, indent=4)
            
        # Move up
        current = parent
