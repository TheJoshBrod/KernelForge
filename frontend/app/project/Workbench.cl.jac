"""Workbench Page"""

import from react { useState, useEffect }
import from "react-router-dom" { useParams, useNavigate }
import from "@jac/runtime" { jacSpawn }
import from "../../components/ui/Icons.cl.jac" {
    CodeIcon,
    CpuIcon,
    ShieldCheckIcon,
    BackIcon,
    ActivityIcon,
    CheckIcon,
    AlertCircleIcon,
    ArrowRightIcon
}


def:pub Workbench {
    params = useParams();
    projectName = params.projectName;
    opName = params.opName;
    navigate = useNavigate();

    tree_state = useState({"nodes": [], "edges": []});
    tree = tree_state[0];
    setTree = tree_state[1];

    selected_node_idx = useState(0);
    selectedIdx = selected_node_idx[0];
    setSelectedIdx = selected_node_idx[1];

    details_state = useState({});
    details = details_state[0];
    setDetails = details_state[1];

    loading_state = useState(True);
    loading = loading_state[0];
    setLoading = loading_state[1];

    async def fetchData -> None {
        if not projectName or not opName { return; }
        setLoading(True);
        try {
            # Fetch Tree
            tree_resp = await jacSpawn("GetMctsTree", "", {"projectName": projectName, "opName": opName});
            tree_data = tree_resp.reports[0] if tree_resp and tree_resp.reports else {};
            if tree_data and "nodes" in tree_data and tree_data["nodes"].length > 0 {
                setTree(tree_data);
                setSelectedIdx(0);
            } else {
                # Filler Data for Demonstration
                filler = {
                    "nodes": [
                        {
                            "id": 1, 
                            "visits": 150, 
                            "value": 0.85, 
                            "best_subtree_value": 0.92, 
                            "improvement_description": "Root state: Initial Triton implementation with default tiling.",
                            "code": "import triton\nimport triton.language as tl\n\n@triton.jit\ndef kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):\n    pid = tl.program_id(0)\n    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n    mask = offsets < 1024\n    x = tl.load(x_ptr + offsets, mask=mask)\n    tl.store(y_ptr + offsets, x, mask=mask)",
                            "timestamp": 1707782400
                        },
                        {
                            "id": 2, 
                            "visits": 45, 
                            "value": 0.88, 
                            "best_subtree_value": 0.88, 
                            "improvement_description": "Increased block size to 256 for better occupancy.",
                            "code": "# Optimized variant with BLOCK_SIZE=256\n@triton.jit\ndef kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):\n    # ... optimized logic ...",
                            "timestamp": 1707782500
                        },
                        {
                            "id": 3, 
                            "visits": 12, 
                            "value": 0.72, 
                            "best_subtree_value": 0.72, 
                            "improvement_description": "Trying warp-level shuffle for reduction (failed to improve throughput).",
                            "code": "# Experimental warp shuffle...",
                            "timestamp": 1707782600
                        }
                    ]
                };
                setTree(filler);
                setSelectedIdx(0);
            }

            # Fetch Details
            details_resp = await jacSpawn("GetOpDetails", "", {"projectName": projectName, "opName": opName});
            details_data = details_resp.reports[0] if details_resp and details_resp.reports else {};
            setDetails(details_data);

        } except (error) {
            console.error("Failed to fetch workbench data:", error);
        } finally {
            setLoading(False);
        }
    }

    useEffect(lambda -> None { fetchData(); }, [projectName, opName]);

    mcts_nodes = tree["nodes"] if "nodes" in tree else [];
    selectedMctsNode = mcts_nodes[selectedIdx] if mcts_nodes.length > selectedIdx else {};

    # Formatter for op name
    def format_op(raw: str) -> str {
        raw_text = raw or "";
        prefix = "torch_nn_functional_";
        if raw_text.startswith(prefix) {
            return "torch.nn.functional." + raw_text.replace(prefix, "");
        }
        return raw_text.replace("_", ".");
    }

    # Render Node List (Timeline-ish)
    node_items = [];
    n_idx = 0;
    while n_idx < mcts_nodes.length {
        mcts_node = mcts_nodes[n_idx];
        is_selected = n_idx == selectedIdx;
        
        # Capture current idx for lambda
        current_idx = n_idx; 

        node_items.append(
            <div 
                key={mcts_node["id"]} 
                className={"p-3 cursor-pointer border-l-2 transition-all " + ("border-indigo-500 bg-indigo-500/5" if is_selected else "border-zinc-800 hover:border-zinc-700 hover:bg-zinc-900/50")}
                onClick={lambda -> None { setSelectedIdx(current_idx); }}
            >
                <div className="flex justify-between items-start mb-1">
                    <span className="text-xs font-mono text-zinc-500">ID: {mcts_node["id"]}</span>
                    <span className="text-[10px] text-zinc-600 bg-zinc-900 px-1 rounded">v{mcts_node["visits"]}</span>
                </div>
                <div className={"text-xs leading-relaxed " + ("text-zinc-200" if is_selected else "text-zinc-400")}>
                    {mcts_node["improvement_description"]}
                </div>
                <div className="mt-1 text-[10px] font-bold text-emerald-500/80">
                    Reward: {(mcts_node["value"] or 0).toFixed(4)}
                </div>
            </div>
        );
        n_idx += 1;
    }

    if loading {
        return <div className="flex-1 flex items-center justify-center bg-zinc-950 text-zinc-500">Loading Op Workbench...</div>;
    }

    if not opName {
        return <div className="flex-1 flex items-center justify-center bg-zinc-950 text-zinc-500">Select an operator from the sidebar to begin exploration.</div>;
    }

    return
        <div className="flex h-screen bg-zinc-950 text-zinc-100 overflow-hidden">
            
            <div className="w-80 border-r border-zinc-800 flex flex-col bg-zinc-950">
                <div className="p-4 border-b border-zinc-800 bg-zinc-900/50 flex justify-between items-center">
                    <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-wider">
                        Search Explorer
                    </h3>
                    <span className="text-[10px] bg-indigo-500/20 text-indigo-400 px-2 py-0.5 rounded-full font-bold">
                        {mcts_nodes.length} Nodes
                    </span>
                </div>
                <div className="overflow-y-auto flex-1 scrollbar-thin scrollbar-thumb-zinc-800">
                    {node_items}
                </div>
            </div>

            
            <div className="flex-1 flex flex-col bg-[#0d0d0d]">
                <div className="h-10 border-b border-zinc-800 flex items-center justify-between px-4 bg-zinc-900/50">
                    <div className="flex items-center gap-2">
                        <span className="text-xs text-zinc-400 font-mono">
                            {format_op(opName)}
                        </span>
                        <span className="text-xs text-zinc-600">/</span>
                        <span className="text-xs text-indigo-400 font-mono">
                            {selectedMctsNode["id"] if "id" in selectedMctsNode else "None"}
                        </span>
                    </div>
                </div>
                <div className="flex-1 p-4 font-mono text-sm overflow-auto text-zinc-300 bg-[#0d0d0d] whitespace-pre">
                    {selectedMctsNode["code"] if "code" in selectedMctsNode else "# No code available for this node"}
                </div>
                
                <div className="h-32 border-t border-zinc-800 bg-zinc-950 p-2 font-mono text-xs overflow-y-auto">
                    <div className="flex items-center gap-2 text-zinc-400 mb-1 border-b border-zinc-800 pb-1">
                        <ShieldCheckIcon className="w-3 h-3" />
                        <span>
                            Node Metrics
                        </span>
                    </div>
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 p-2 text-zinc-400">
                        <div>
                            <div className="text-[10px] text-zinc-500 uppercase">Visits</div>
                            <div className="text-white text-sm">{selectedMctsNode["visits"] or 0}</div>
                        </div>
                        <div>
                            <div className="text-[10px] text-zinc-500 uppercase">Current Value</div>
                            <div className="text-emerald-400 text-sm font-bold">{(selectedMctsNode["value"] or 0).toFixed(6)}</div>
                        </div>
                        <div>
                            <div className="text-[10px] text-zinc-500 uppercase">Best Subtree Value</div>
                            <div className="text-indigo-400 text-sm font-bold">{(selectedMctsNode["best_subtree_value"] or 0).toFixed(6)}</div>
                        </div>
                        <div>
                            <div className="text-[10px] text-zinc-500 uppercase">Timestamp</div>
                            <div className="text-zinc-500 text-xs">
                                {
                                    Reflect.construct(Date, [selectedMctsNode["timestamp"] * 1000]).toLocaleString() 
                                    if selectedMctsNode["timestamp"] 
                                    else "N/A"
                                }
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            
            <div className="w-96 border-l border-zinc-800 bg-zinc-950 flex flex-col">
                <div className="p-3 border-b border-zinc-800 bg-zinc-900/50 flex justify-between items-center">
                    <span className="text-sm font-semibold flex gap-2 items-center text-zinc-200">
                        <CpuIcon className="w-4 h-4" />
                        <span>
                            Kernel Architect
                        </span>
                    </span>
                    <button className="bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-[10px] px-2 py-1 rounded transition-colors" onClick={fetchData}>
                        Sync
                    </button>
                </div>
                <div className="flex-1 p-4 space-y-4 overflow-y-auto bg-zinc-900/20">
                    
                    <div className="flex gap-3">
                        <div className="w-6 h-6 rounded-full bg-indigo-600 flex items-center justify-center text-[10px] text-white font-bold flex-shrink-0">
                            AI
                        </div>
                        <div className="bg-zinc-800 p-3 rounded-lg rounded-tl-none text-sm text-zinc-300 border border-zinc-700/50 shadow-sm">
                            {selectedMctsNode["improvement_description"] if "improvement_description" in selectedMctsNode else "Selecting a node to see the optimization strategy..."}
                        </div>
                    </div>

                    {
                        <div className="mt-4 p-4 border border-zinc-800 rounded bg-zinc-900/30">
                            <h4 className="text-xs font-bold text-zinc-500 uppercase mb-2">History</h4>
                            <div className="text-[10px] text-zinc-400 leading-relaxed">
                                This node was explored during an MCTS iteration focusing on reducing memory overhead.
                            </div>
                        </div>
                        if selectedMctsNode["id"]
                        else <span></span>
                    }
                </div>
                <div className="p-3 border-t border-zinc-800 bg-zinc-900/50">
                    <button className="w-full bg-indigo-600 hover:bg-indigo-500 text-white text-xs py-2 rounded flex justify-center gap-2 items-center font-bold transition-all shadow-lg hover:shadow-indigo-500/20">
                        Deploy this Variant
                    </button>
                </div>
            </div>
        </div>;
}

