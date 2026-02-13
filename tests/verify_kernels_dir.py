
import sys
import shutil
import json
import sqlite3
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd()))

import src.optimizer.core.mcts as mcts
from src.optimizer.core.types import KernelNode

def test_kernels_dir():
    print("Setting up mock project for kernels dir test...")
    proj_dir = Path("tests/mock_kernels_project/trees")
    if proj_dir.exists():
        shutil.rmtree(proj_dir)
    
    # We expect 'kernels' dir to be created by the caller (pipeline usually), 
    # but let's see if save_node expects it to exist or if it just writes DB.
    # save_node calls update_tree, which writes to DB.
    # The FILE writing happens in pipeline.py options, NOT in save_node anymore?
    # Wait, save_iteration in pipeline.py writes the file.
    
    (proj_dir / "kernels").mkdir(parents=True)
    
    paths = {"proj_dir": proj_dir}
    mcts.init_db(paths)
    
    # Check NO nodes dir
    if (proj_dir / "nodes").exists():
        print("FAIL: 'nodes' directory created by init_db?")
    else:
        print("PASS: 'nodes' directory not created")
        
    # Create a new node
    node = KernelNode(
        id=1,
        parent_id=-1,
        children_ids=[],
        visits=1,
        value=50.0,
        best_subtree_value=50.0,
        code=str(proj_dir / "kernels" / "kernel_1.cu"),
        improvement_description="Test Rename"
    )
    
    print("Saving node 1...")
    mcts.save_node(paths, node)
    
    # Check DB
    db_path = proj_dir / "nodes.db"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT code FROM nodes WHERE id=1")
        row = cursor.fetchone()
        if row and "kernels/kernel_1.cu" in row[0]:
            print("PASS: DB contains correct path with 'kernels'")
        else:
            print(f"FAIL: DB path mismatch. Row: {row}")

if __name__ == "__main__":
    try:
        test_kernels_dir()
    finally:
        if Path("tests/mock_kernels_project").exists():
            shutil.rmtree("tests/mock_kernels_project")
