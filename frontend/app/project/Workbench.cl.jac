"""Workbench Page"""

import from react { useState, useEffect, useCallback }
import from "react-router-dom" { useParams, useNavigate }
import from "@jac/runtime" { jacSpawn }
import from "reactflow" { 
    ReactFlow, 
    Background, 
    Controls, 
    MiniMap,
    useNodesState,
    useEdgesState,
    MarkerType
}
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

    # ReactFlow State
    nodes_state = useNodesState([]);
    nodes = nodes_state[0];
    setNodes = nodes_state[1];
    onNodesChange = nodes_state[2];

    edges_state = useEdgesState([]);
    edges = edges_state[0];
    setEdges = edges_state[1];
    onEdgesChange = edges_state[2];
    
    # Selection State
    selected_node_id_state = useState(None);
    selectedNodeId = selected_node_id_state[0];
    setSelectedNodeId = selected_node_id_state[1];

    # Data State
    raw_nodes_state = useState({}); # Map ID -> Node Data
    rawNodes = raw_nodes_state[0];
    setRawNodes = raw_nodes_state[1];

    loading_state = useState(True);
    loading = loading_state[0];
    setLoading = loading_state[1];

    ops_state = useState([]);
    ops = ops_state[0];
    setOps = ops_state[1];

    loading_ops_state = useState(True);
    loadingOps = loading_ops_state[0];
    setLoadingOps = loading_ops_state[1];

    # View Mode State
    view_mode_state = useState("graph"); # "graph" or "list"
    viewMode = view_mode_state[0];
    setViewMode = view_mode_state[1];

    # --- Helpers ---

    def get_layouted_elements(nodes_data: list, edges_data: list) -> dict {
        # Simple tree layout: 
        # Organize by depth/levels.
        # Root is level 0. Children are level 1, etc.
        # X position based on index in level.
        
        levels = {};
        node_map = {};
        children_map = {};

        # Build maps
        for n in nodes_data {
            node_map[n["id"]] = n;
            children_map[n["id"]] = [];
        }

        # Identify parent/child relationships
        has_parent = {};
        for e in edges_data {
            pid = e["parent_id"];
            cid = e["child_id"];
            if pid in children_map {
                children_map[pid].push(cid);
            }
            has_parent[cid] = True;
        }

        # Find roots
        roots = [];
        for n in nodes_data {
            if not n["id"] in has_parent {
                roots.push(n["id"]);
            }
        }

        # BFS for levels
        queue = [];
        for r in roots {
            queue.push({"id": r, "level": 0});
        }
        
        layouted_nodes = [];
        visited = {};
        
        level_counts = {}; # Track how many nodes in each level for X positioning
        max_level = 0;

        while queue.length > 0 {
            curr = queue.shift();
            cid = curr["id"];
            lvl = curr["level"];
            
            if lvl > max_level {
                max_level = lvl;
            }

            if cid in visited {
                continue;
            }
            visited[cid] = True;

            # Assign position
            if not lvl in level_counts {
                level_counts[lvl] = 0;
            }
            
            x_pos = level_counts[lvl] * 200; # 200px spacing
            y_pos = lvl * 150; # 150px spacing
            level_counts[lvl] += 1;

            # Create ReactFlow Node
            raw_node = node_map[cid];
            is_best = False; # visual hint?
            
            # Label
            label_val = (raw_node["value"] or 0).toFixed(4);
            
            layouted_nodes.push({
                "id": f"{cid}",
                "position": {"x": x_pos, "y": y_pos},
                "data": {"label": f"Kernel_{cid}\n{label_val} ms"},
                "style": {
                    "background": "#18181b", # zinc-900
                    "color": "#e4e4e7", # zinc-200
                    "border": "1px solid #3f3f46", # zinc-700
                    "borderRadius": "8px",
                    "padding": "10px",
                    "fontSize": "12px",
                    "fontFamily": "monospace",
                    "width": 100,
                    "textAlign": "center"
                },
                "type": "default"
            });

            # Add children to queue
            children = children_map[cid] if cid in children_map else [];
            for child_id in children {
                queue.push({"id": child_id, "level": lvl + 1});
            }
        }

        # Center align levels (heuristic)
        # Calculate max width of a level
        max_width = 0;
        itr_l = 0;
        while itr_l <= max_level {
            cnt = level_counts[itr_l] if itr_l in level_counts else 0;
            w = cnt * 200;
            if w > max_width {
                max_width = w;
            }
            itr_l += 1;
        }
        
        # Adjust X to center
        final_nodes = [];
        for n in layouted_nodes {
            # Recalculate based on row width? 
            # For simplicity, keeping left-aligned for now as tree grows right/down usually
            # But converting to standard format
            final_nodes.push(n);
        }

        # Edges
        rf_edges = [];
        for e in edges_data {
            rf_edges.push({
                "id": f"e{e['parent_id']}-{e['child_id']}",
                "source": f"{e['parent_id']}",
                "target": f"{e['child_id']}",
                "type": "smoothstep",
                "animated": True,
                "style": { "stroke": "#6366f1" } # indigo-500
            });
        }

        return {"nodes": final_nodes, "edges": rf_edges};
    }

    # --- Fetchers ---

    async def fetchOps -> None {
        setLoadingOps(True);
        try {
            ops_resp = await jacSpawn("GetProjectOps", "", {"projectName": projectName});
            ops_data = ops_resp.reports[0] if ops_resp and ops_resp.reports else None;
            op_list = ops_data["ops"] if ops_data and "ops" in ops_data else [];
            setOps(op_list);
        } except Exception as e {
            console.error("Failed to fetch ops:", e);
        } finally {
            setLoadingOps(False);
        }
    }

    async def fetchData -> None {
        if not projectName or not opName { 
            setLoading(False);
            return; 
        }
        setLoading(True);
        try {
            tree_resp = await jacSpawn("GetMctsTree", "", {"projectName": projectName, "opName": opName});
            tree_data = tree_resp.reports[0] if tree_resp and tree_resp.reports else {};
            
            m_nodes = tree_data["nodes"] if tree_data and "nodes" in tree_data else [];
            m_edges = tree_data["edges"] if tree_data and "edges" in tree_data else [];

            layout = get_layouted_elements(m_nodes, m_edges);
            
            setNodes(layout["nodes"]);
            setEdges(layout["edges"]);

            # Populate Raw Map
            r_map = {};
            best_node_id = None;
            max_val = -1;

            for n in m_nodes {
                r_map[f"{n['id']}"] = n;
                val = n["value"] or 0;
                if val > max_val {
                    max_val = val;
                    best_node_id = f"{n['id']}";
                }
            }
            setRawNodes(r_map);
            
            # Default select best node
            if best_node_id {
                setSelectedNodeId(best_node_id);
            }

        } except Exception as e {
            console.error("Failed to fetch workbench data:", e);
        } finally {
            setLoading(False);
        }
    }

    useEffect(lambda -> None { if not opName { fetchOps(); } }, [opName, projectName]);
    useEffect(lambda -> None { fetchData(); }, [projectName, opName]);

    # --- Actions ---

    onNodeClick = useCallback(lambda event: any, node: any -> None {
        setSelectedNodeId(node.id);
    }, []);

    # Highlighting Effect
    useEffect(lambda -> None {
        setNodes(lambda nds: list -> list {
            new_nodes = [];
            for n in nds {
                new_style = {
                    "background": "#18181b", 
                    "color": "#e4e4e7", 
                    "border": "1px solid #3f3f46",
                    "borderRadius": "8px",
                    "padding": "10px",
                    "fontSize": "12px",
                    "fontFamily": "monospace",
                    "width": 100,
                    "textAlign": "center"
                };
                if n.id == selectedNodeId {
                    new_style["border"] = "2px solid #6366f1"; # indigo-500
                    new_style["background"] = "rgba(99, 102, 241, 0.1)"; # indigo tint
                    new_style["boxShadow"] = "0 0 10px rgba(99, 102, 241, 0.3)";
                }
                new_nodes.push({
                    "id": n.id,
                    "position": n.position,
                    "data": n.data,
                    "style": new_style,
                    "type": n.type
                });
            }
            return new_nodes;
        });
    }, [selectedNodeId, setNodes]);

    async def handleUpload -> None {
        try {
            console.log("Uploading custom kernel...");
            resp = await jacSpawn("ImportKernel", "", {});
            console.log("Upload result:", resp);
            alert("Custom kernel uploaded!"); # Simple feedback
        } except (e) {
            console.error("Upload failed", e);
        }
    }

    # --- Derived State ---
    
    selectedNodeData = rawNodes[selectedNodeId] if selectedNodeId and selectedNodeId in rawNodes else {};

    # Sorted Nodes for List View
    sortedNodes = [];
    if rawNodes {
        # Convert map to list
        vals = Object.values(rawNodes);
        for v in vals {
            sortedNodes.push(v);
        }
        # Sort by value descending
        sortedNodes.sort(lambda a: any, b: any -> int {
            return (b["value"] or 0) - (a["value"] or 0);
        });
    }

    # Generate List View Items
    listViewItems = [];
    if viewMode == "list" {
        idx = 0;
        for n in sortedNodes {
            idx += 1;
            listViewItems.push(
                <div 
                    key={n["id"]}
                    className={"p-3 rounded-lg border cursor-pointer transition-all flex items-center gap-4 group " + ("border-indigo-500 bg-indigo-500/10 shadow-[0_0_15px_rgba(99,102,241,0.15)]" if f"{n['id']}" == selectedNodeId else "border-zinc-800 bg-zinc-900/30 hover:border-zinc-700 hover:bg-zinc-900/80")}
                    onClick={lambda -> None { setSelectedNodeId(f"{n['id']}"); }}
                >
                    <div className="text-zinc-500 font-mono text-xs w-8 text-right">
                        {"#" + idx}
                    </div>

                    <div className={"font-mono text-sm font-bold " + ("text-indigo-300" if f"{n['id']}" == selectedNodeId else "text-zinc-300")}>
                        Kernel_{n["id"]}:
                    </div>

                    <div className="font-mono text-emerald-400 text-sm font-bold flex-1">
                        {(n["value"] or 0).toFixed(6) + " ms"}
                    </div>

                    <div className="text-zinc-600 text-xs font-mono w-40 text-right">
                        {
                            Reflect.construct(Date, [n["timestamp"] * 1000]).toLocaleString() 
                            if "timestamp" in n 
                            else ""
                        }
                    </div>
                </div>
            );
        }
    }

    # Iteration Log Logic (Ancestry)
    log_items = [];
    if selectedNodeId and nodes.length > 0 and edges.length > 0 {
        # Build parent map
        parent_map = {};
        for e in edges {
            parent_map[e.target] = e.source;
        }

        curr = selectedNodeId;
        while curr {
            n_data = rawNodes[curr];
            if n_data {
                log_items.unshift(n_data); # Add to front to make Root -> Leaf
            }
            curr = parent_map[curr] if curr in parent_map else None;
        }
    }

    # Format Op Name
    def format_op(raw: str) -> str {
        raw_text = raw or "";
        prefix = "torch_nn_functional_";
        if raw_text.startswith(prefix) {
            return "torch.nn.functional." + raw_text.replace(prefix, "");
        }
        return raw_text.replace("_", ".");
    }

    # --- Renders ---

    if loading {
        return <div className="flex items-center justify-center h-screen bg-zinc-950 text-zinc-500">Loading Workbench...</div>;
    }

    if not opName {
        # (Keep existing Op Selector Logic)
        op_items = [];
        for op in ops {
            op_items.push(
                <div 
                    key={op["name"]}
                    className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg hover:border-indigo-500/50 cursor-pointer transition-all flex justify-between items-center group"
                    onClick={lambda -> None { navigate(f"/project/{projectName}/workbench/{op['name']}"); }}
                >
                    <span className="font-mono text-sm text-zinc-300 group-hover:text-white">{format_op(op["name"])}</span>
                    <span className="text-xs text-zinc-600 bg-zinc-950 px-2 py-1 rounded">{op["count"]} {" calls"}</span>
                </div>
            );
        }
        return 
            <div className="flex-1 flex flex-col p-8 bg-zinc-900 text-zinc-100 h-screen overflow-hidden">
                <div className="mb-8">
                    <h2 className="text-2xl font-bold text-white mb-2">Workbench</h2>
                    <p className="text-zinc-400">Select an operator to explore.</p>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 overflow-y-auto pb-20">
                    {op_items}
                    {
                        <div className="col-span-full text-center text-zinc-500 py-12 italic">
                            Loading operators...
                        </div>
                        if loadingOps
                        else
                        <div className="col-span-full text-center text-zinc-500 py-12 italic">
                            No operators found.
                        </div>
                        if ops.length == 0
                        else <span></span>
                    }
                </div>
            </div>;
    }

    # --- Main Split Layout ---

    return
        <div className="flex h-screen w-full bg-zinc-950 text-zinc-100 overflow-hidden">
            
            # Left Panel: Code (Top) + Log (Bottom)
            <div className="w-1/2 flex flex-col border-r border-zinc-800">
                
                # Code Section (Flex grow)
                <div className="flex-1 flex flex-col min-h-0 border-b border-zinc-800">
                    <div className="h-10 bg-zinc-900/50 border-b border-zinc-800 flex items-center px-4 justify-between">
                        <span className="text-xs font-bold text-zinc-500 uppercase flex items-center gap-2">
                            <CodeIcon className="w-4 h-4" /> Kernel Code
                        </span>
                        <span className="text-xs font-mono text-indigo-400">
                            Node {selectedNodeId or "?"}
                        </span>
                    </div>
                    <div className="flex-1 overflow-auto bg-[#0d0d0d] p-4">
                        <pre className="font-mono text-xs text-zinc-300 whitespace-pre">
                            {selectedNodeData["code"] if "code" in selectedNodeData else "// Select a node to view code"}
                        </pre>
                    </div>
                </div>

                # Log Section (Fixed height or flex?) - "Below that... entire log... scrollable but start at selected node on bottom"
                # Let's make it 40% height
                <div className="h-[40%] flex flex-col bg-zinc-900/20">
                    <div className="h-10 bg-zinc-900/50 border-b border-zinc-800 flex items-center px-4">
                        <span className="text-xs font-bold text-zinc-500 uppercase flex items-center gap-2">
                            <ActivityIcon className="w-4 h-4" /> Iteration Log
                        </span>
                    </div>
                    <div className="flex-1 overflow-y-auto p-4 space-y-2 flex flex-col-reverse"> 
                        # flex-col-reverse to have scroll start at bottom? No, typically "scroll to bottom" means the latest item is at the bottom and view is there.
                        # Standard list, iterate normal order.
                        
                        {
                        {
                            [
                                <div 
                                    key={item["id"]}
                                    className={"p-3 rounded border text-xs " + ("border-indigo-500/50 bg-indigo-500/10" if f"{item['id']}" == selectedNodeId else "border-zinc-800 bg-zinc-900/50")}
                                    ref={lambda el: any -> None { 
                                        if f"{item['id']}" == selectedNodeId and el { 
                                            el.scrollIntoView({"behavior": "smooth", "block": "nearest"});
                                        }
                                    }}
                                >
                                    <div className="flex justify-between mb-1">
                                        <span className="font-mono font-bold text-indigo-300">kernel_{item["id"]}</span>
                                        <span className="text-emerald-400 font-mono">{(item["value"] or 0).toFixed(4)}</span>
                                    </div>
                                    <div className="text-zinc-400">
                                        {item["improvement_description"]}
                                    </div>
                                </div>
                                for item in log_items
                            ]
                        }
                        }
                    </div>
                </div>
            </div>

            # Right Panel: Tree (Top) + Info (Middle) + Button (Bottom)
            <div className="w-1/2 flex flex-col bg-zinc-950">
                
                # Tree Viz (Top, 50%)
                # Tree Viz (Top, 50%)
                <div className="h-[50%] border-b border-zinc-800 relative bg-[#09090b]">
                    # View Toggle
                    <div className="absolute top-2 left-2 z-10 bg-zinc-900/90 rounded-lg border border-zinc-800 p-1 flex gap-1 shadow-lg">
                        <button 
                            className={"px-3 py-1.5 text-xs font-bold rounded transition-all " + ("bg-zinc-700 text-white shadow-sm" if viewMode == "graph" else "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50")}
                            onClick={lambda: setViewMode("graph")}
                        >
                            Graph
                        </button>
                        <button 
                            className={"px-3 py-1.5 text-xs font-bold rounded transition-all " + ("bg-zinc-700 text-white shadow-sm" if viewMode == "list" else "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50")}
                            onClick={lambda: setViewMode("list")}
                        >
                            List
                        </button>
                    </div>

                    {
                        <ReactFlow
                            nodes={nodes}
                            edges={edges}
                            onNodesChange={onNodesChange}
                            onEdgesChange={onEdgesChange}
                            onNodeClick={onNodeClick}
                            fitView={True}
                            minZoom={0.1}
                            nodesDraggable={False}
                            proOptions={{"hideAttribution": True}}
                        >
                            <Background color="#27272a" gap={20} />
                        </ReactFlow>
                        if viewMode == "graph"
                        else
                        <div className="h-full overflow-y-auto p-4 pt-14 space-y-2">
                            {listViewItems}
                            {
                                <div className="text-center py-10 text-zinc-600 italic text-sm">
                                    No kernels found.
                                </div>
                                if sortedNodes.length == 0
                                else <span />
                            }
                        </div>
                    }
                </div>

                # Info Panel (Middle, Flex)
                <div className="flex-1 p-6 overflow-y-auto">
                    <div className="flex justify-between items-center mb-4 border-b border-zinc-800 pb-2">
                        <h3 className="text-lg font-bold text-white flex items-center gap-2">
                            <ShieldCheckIcon className="w-5 h-5 text-emerald-500" />    
                            Node Details
                        </h3>
                        <span className="font-mono text-zinc-400 text-xs">
                            {
                                Reflect.construct(Date, [selectedNodeData["timestamp"] * 1000]).toLocaleString() 
                                if "timestamp" in selectedNodeData 
                                else ""
                            }
                        </span>
                    </div>
                    {
                        <div className="space-y-4">
                            <div className="flex justify-between items-center">
                                <span className="text-zinc-500 text-sm">Node ID</span>
                                <span className="font-mono text-indigo-400">{selectedNodeData["id"]}</span>
                            </div>
                            <div className="flex justify-between items-center">
                                <span className="text-zinc-500 text-sm">Visits</span>
                                <span className="font-mono text-white">{selectedNodeData["visits"]}</span>
                            </div>
                            <div className="flex justify-between items-center">
                                <span className="text-zinc-500 text-sm">Reward</span>
                                <span className="font-mono text-emerald-400">{(selectedNodeData["value"] or 0).toFixed(6)}</span>
                            </div>
                            <div className="flex justify-between items-center">
                                <span className="text-zinc-500 text-sm">Best Subtree</span>
                                <span className="font-mono text-purple-400">{(selectedNodeData["best_subtree_value"] or 0).toFixed(6)}</span>
                            </div>
                        </div>
                        if selectedNodeId 
                        else <div className="text-zinc-500 text-sm italic">Select a node from the tree to view details.</div>
                    }

                </div>

                # Upload Button (Bottom)
                <div className="p-4 border-t border-zinc-800 bg-zinc-900/30">
                    <button 
                        className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-3 px-4 rounded-lg transition-all shadow-lg hover:shadow-indigo-500/20 flex items-center justify-center gap-2"
                        onClick={handleUpload}
                    >
                        <CpuIcon className="w-5 h-5" />
                        Upload New Kernel
                    </button>
                    <div className="text-center mt-2 text-[10px] text-zinc-500">
                        Uploads the current kernel as a new root for optimization.
                    </div>
                </div>

            </div>
        </div>;
}
