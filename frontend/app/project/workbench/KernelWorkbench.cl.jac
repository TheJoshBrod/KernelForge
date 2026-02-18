"""Kernel Workbench Component"""

import from react { useState, useEffect, useCallback }
import from "react-router-dom" { useNavigate, useParams }
import from "reactflow" {
    ReactFlow,
    Background,
    Controls,
    MiniMap,
    useNodesState,
    useEdgesState,
    MarkerType
}
import from ....components.ui.Icons {
    CodeIcon,
    CpuIcon,
    ShieldCheckIcon,
    BackIcon,
    ActivityIcon,
    CheckIcon,
    AlertCircleIcon,
    ArrowRightIcon
}
import from .NewKernel { NewKernel }

def:pub KernelWorkbench {
    has projectName: str,
        opName: str;

    navigate = useNavigate();
    params = useParams();
    resolvedProjectName = projectName or params.projectName;
    resolvedOpName = opName or params.opName;

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
    raw_nodes_state = useState({});  # Map ID -> Node Data
    rawNodes = raw_nodes_state[0];
    setRawNodes = raw_nodes_state[1];

    loading_state = useState(True);
    loading = loading_state[0];
    setLoading = loading_state[1];

    # View Mode State
    view_mode_state = useState("graph");  # "graph" or "list"
    viewMode = view_mode_state[0];
    setViewMode = view_mode_state[1];

    # --- Helpers ---
    def get_layouted_elements(nodes_data: list, edges_data: list) -> dict {
        levels = {};
        node_map = {};
        children_map = {};

        for n in nodes_data {
            node_map[n["id"]] = n;
            children_map[n["id"]] = [];
        }

        has_parent = {};
        for e in edges_data {
            pid = e["parent_id"];
            cid = e["child_id"];
            if pid in children_map {
                children_map[pid].push(cid);
            }
            has_parent[cid] = True;
        }

        roots = [];
        for n in nodes_data {
            if not n["id"] in has_parent {
                roots.push(n["id"]);
            }
        }

        queue = [];
        for r in roots {
            queue.push({"id": r, "level": 0});
        }

        layouted_nodes = [];
        visited = {};
        level_counts = {};
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

            if not lvl in level_counts {
                level_counts[lvl] = 0;
            }

            x_pos = level_counts[lvl] * 200;
            y_pos = lvl * 150;
            level_counts[lvl] += 1;

            raw_node = node_map[cid];
            label_val = Number(raw_node["value"] or 0).toFixed(4);

            layouted_nodes.push(
                {
                    "id": f"{cid}",
                    "position": {"x": x_pos, "y": y_pos},
                    "data": {"label": f"Kernel_{cid}\n{label_val} ms"},
                    "style": {
                        "background": "#18181b",
                        "color": "#e4e4e7",
                        "border": "1px solid #3f3f46",
                        "borderRadius": "8px",
                        "padding": "10px",
                        "fontSize": "12px",
                        "fontFamily": "monospace",
                        "width": 100,
                        "textAlign": "center"
                    },
                    "type": "default"
                }
            );

            children = children_map[cid] if cid in children_map else [];
            for child_id in children {
                queue.push({"id": child_id, "level": lvl + 1});
            }
        }

        rf_edges = [];
        for e in edges_data {
            rf_edges.push(
                {
                    "id": f"e{e['parent_id']}-{e['child_id']}",
                    "source": f"{e['parent_id']}",
                    "target": f"{e['child_id']}",
                    "type": "smoothstep",
                    "animated": True,
                    "style": {"stroke": "#6366f1"}
                }
            );
        }

        return {"nodes": layouted_nodes, "edges": rf_edges};
    }

    async def fetchData -> None {
        if not resolvedProjectName or not resolvedOpName {
            setLoading(False);
            return;
        }
        setLoading(True);
        try {
            tree_resp = await __jacSpawn(
                "GetMctsTree",
                "",
                {"projectName": resolvedProjectName, "opName": resolvedOpName}
            );
            tree_data = tree_resp.reports[0] if tree_resp and tree_resp.reports else {};

            m_nodes = tree_data["nodes"] if tree_data and "nodes" in tree_data else [];
            m_edges = tree_data["edges"] if tree_data and "edges" in tree_data else [];

            layout = get_layouted_elements(m_nodes, m_edges);

            setNodes(layout["nodes"]);
            setEdges(layout["edges"]);

            r_map = {};
            best_node_id = None;
            best_val = None;

            for n in m_nodes {
                r_map[f"{n['id']}"] = n;
                val = Number(n["value"]) if n and "value" in n else 0;
                if best_val is None or val < best_val {
                    best_val = val;
                    best_node_id = f"{n['id']}";
                }
            }
            setRawNodes(r_map);

            if best_node_id {
                setSelectedNodeId(best_node_id);
            }
        } except Exception as e {
            console.error("Failed to fetch workbench data:", e);
        } finally {
            setLoading(False);
        }
    }

    useEffect(lambda -> None { fetchData();}, [resolvedProjectName, resolvedOpName]);

    onNodeClick = useCallback(
        lambda event: any , node: any  -> None { setSelectedNodeId(node.id);}, []
    );

    useEffect(
        lambda -> None { setNodes(
            lambda nds: list  -> list { new_nodes = [];for n in nds {
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
                    new_style["border"] = "2px solid #6366f1";
                    new_style["background"] = "rgba(99, 102, 241, 0.1)";
                    new_style["boxShadow"] = "0 0 10px rgba(99, 102, 241, 0.3)";
                }
                new_nodes.push(
                    {
                        "id": n.id,
                        "position": n.position,
                        "data": n.data,
                        "style": new_style,
                        "type": n.type
                    }
                );
            }return new_nodes; }
        );},
        [selectedNodeId, setNodes]
    );

    selectedNodeData = rawNodes[selectedNodeId]
    if selectedNodeId and selectedNodeId in rawNodes
    else {};

    sortedNodes = [];
    if rawNodes {
        vals = Object.values(rawNodes);
        for v in vals {
            sortedNodes.push(v);
        }
        sortedNodes.sort(
            lambda a: any, b: any -> float {
                val_a = Number(a["value"]) if a and "value" in a else 0;
                val_b = Number(b["value"]) if b and "value" in b else 0;
                return val_a - val_b;
            }
        );
    }

    listViewItems = [];
    if viewMode == "list" {
        idx = 0;
        for n in sortedNodes {
            idx += 1;
            listViewItems.push(
                <div
                    key={n["id"]}
                    className={"p-3 rounded-lg border cursor-pointer transition-all flex items-center gap-4 group " + (
                        "border-indigo-500 bg-indigo-500/10 shadow-[0_0_15px_rgba(99,102,241,0.15)]"
                        if f"{n['id']}" == selectedNodeId
                        else "border-zinc-800 bg-zinc-900/30 hover:border-zinc-700 hover:bg-zinc-900/80"
                    )}
                    onClick={lambda -> None { setSelectedNodeId(f"{n['id']}");}}
                >
                    <div className="text-zinc-500 font-mono text-xs w-8 text-right">
                        {"#" + str(idx)}
                    </div>
                    <div
                        className={"font-mono text-sm font-bold " + (
                            "text-indigo-300"
                            if f"{n['id']}" == selectedNodeId
                            else "text-zinc-300"
                        )}
                    >
                        Kernel_{n["id"]}:
                    </div>
                    <div
                        className="font-mono text-emerald-400 text-sm font-bold flex-1"
                    >
                        <span>
                            {Number(n["value"] or 0).toFixed(6)}
                        </span>
                        <span>
                            {" ms"}
                        </span>
                    </div>
                    <div className="text-zinc-600 text-xs font-mono w-40 text-right">
                        {Reflect.construct(Date, [n["timestamp"] * 1000]).toLocaleString()
                        if "timestamp" in n
                        else ""}
                    </div>
                </div>
            );
        }
    }

    log_items = [];
    if selectedNodeId and nodes.length > 0 and edges.length > 0 {
        parent_map = {};
        for e in edges {
            parent_map[e.target] = e.source;
        }
        curr = selectedNodeId;
        while curr {
            n_data = rawNodes[curr];
            if n_data {
                log_items.unshift(n_data);
            }
            curr = parent_map[curr] if curr in parent_map else None;
        }
    }

    logViewItems = [];
    for item in log_items {
        logViewItems.push(
            <div
                key={item["id"]}
                className={"p-3 rounded border text-xs " + (
                    "border-indigo-500/50 bg-indigo-500/10"
                    if f"{item['id']}" == selectedNodeId
                    else "border-zinc-800 bg-zinc-900/50"
                )}
                ref={lambda el: any  -> None { if f"{item['id']}" == selectedNodeId
                and el {
                    el.scrollIntoView({"behavior": "smooth", "block": "nearest"});
                }}}
            >
                <div className="flex justify-between mb-1">
                    <span className="font-mono font-bold text-indigo-300">
                        kernel_{item["id"]}
                    </span>
                    <span className="text-emerald-400 font-mono">
                        {Number(item["value"] or 0).toFixed(4)}
                    </span>
                </div>
                <div className="text-zinc-400">
                    {item["improvement_description"]}
                </div>
            </div>
        );
    }

    if loading {
        return
            <div
                className="flex items-center justify-center h-screen bg-zinc-950 text-zinc-500"
            >
                Loading Workbench...
            </div>;
    }

    return
        <div
            className="flex h-screen w-full bg-zinc-950 text-zinc-100 overflow-hidden"
        >
            <div className="w-1/2 flex flex-col border-r border-zinc-800">
                <div className="flex-1 flex flex-col min-h-0 border-b border-zinc-800">
                    <div
                        className="h-10 bg-zinc-900/50 border-b border-zinc-800 flex items-center px-4 justify-between"
                    >
                        <span
                            className="text-xs font-bold text-zinc-500 uppercase flex items-center gap-2"
                        >
                            <CodeIcon className="w-4 h-4" />
                            Kernel Code
                        </span>
                        <span className="text-xs font-mono text-indigo-400">
                            Node {selectedNodeId or "?"}
                        </span>
                    </div>
                    <div className="flex-1 overflow-auto bg-[#0d0d0d] p-4">
                        <pre
                            className="font-mono text-xs text-zinc-300 whitespace-pre"
                        >
                            {selectedNodeData["code"]
                            if "code" in selectedNodeData
                            else "// Select a node to view code"}
                        </pre>
                    </div>
                </div>
                <div className="h-[40%] flex flex-col bg-zinc-900/20">
                    <div
                        className="h-10 bg-zinc-900/50 border-b border-zinc-800 flex items-center px-4"
                    >
                        <span
                            className="text-xs font-bold text-zinc-500 uppercase flex items-center gap-2"
                        >
                            <ActivityIcon className="w-4 h-4" />
                            Iteration Log
                        </span>
                    </div>
                    <div
                        className="flex-1 overflow-y-auto p-4 space-y-2 flex flex-col-reverse"
                    >
                        {logViewItems}
                    </div>
                </div>
            </div>
            <div className="w-1/2 flex flex-col bg-zinc-950">
                <div
                    className="h-[50%] border-b border-zinc-800 relative bg-[#09090b]"
                >
                    <div
                        className="absolute top-2 left-2 z-10 bg-zinc-900/90 rounded-lg border border-zinc-800 p-1 flex gap-1 shadow-lg"
                    >
                        <button
                            className={"px-3 py-1.5 text-xs font-bold rounded transition-all " + (
                                "bg-zinc-700 text-white shadow-sm"
                                if viewMode == "graph"
                                else "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50"
                            )}
                            onClick={lambda : setViewMode("graph")}
                        >
                            Graph
                        </button>
                        <button
                            className={"px-3 py-1.5 text-xs font-bold rounded transition-all " + (
                                "bg-zinc-700 text-white shadow-sm"
                                if viewMode == "list"
                                else "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50"
                            )}
                            onClick={lambda : setViewMode("list")}
                        >
                            List
                        </button>
                    </div>
                    {<ReactFlow
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
                    else <div className="h-full overflow-y-auto p-4 pt-14 space-y-2">
                        {listViewItems}{<div
                            className="text-center py-10 text-zinc-600 italic text-sm"
                        >
                            No kernels found.
                        </div>
                        if sortedNodes.length == 0
                        else <span />}
                    </div>}
                </div>
                <div className="flex-1 p-6 overflow-y-auto">
                    <div
                        className="flex justify-between items-center mb-4 border-b border-zinc-800 pb-2"
                    >
                        <h3
                            className="text-lg font-bold text-white flex items-center gap-2"
                        >
                            <ShieldCheckIcon className="w-5 h-5 text-emerald-500" />
                            Node Details
                        </h3>
                        <span className="font-mono text-zinc-400 text-xs">
                            {Reflect.construct(
                                Date, [selectedNodeData["timestamp"] * 1000]
                            ).toLocaleString()
                            if "timestamp" in selectedNodeData
                            else ""}
                        </span>
                    </div>
                    {<div className="space-y-4">
                        <div className="flex justify-between items-center">
                            <span className="text-zinc-500 text-sm">
                                Node ID
                            </span>
                            <span className="font-mono text-indigo-400">
                                {selectedNodeData["id"]}
                            </span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-zinc-500 text-sm">
                                Visits
                            </span>
                            <span className="font-mono text-white">
                                {selectedNodeData["visits"]}
                            </span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-zinc-500 text-sm">
                                Reward
                            </span>
                            <span className="font-mono text-emerald-400">
                                {Number(selectedNodeData["value"] or 0).toFixed(6)}
                            </span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-zinc-500 text-sm">
                                Best Subtree
                            </span>
                            <span className="font-mono text-purple-400">
                                {Number(selectedNodeData["best_subtree_value"] or 0).toFixed(
                                    6
                                )}
                            </span>
                        </div>
                    </div>
                    if selectedNodeId
                    else <div className="text-zinc-500 text-sm italic">
                        Select a node from the tree to view details.
                    </div>}
                </div>
                <div className="p-4 border-t border-zinc-800 bg-zinc-900/30">
                    <NewKernel
                        projectName={resolvedProjectName}
                        opName={resolvedOpName}
                    />
                </div>
            </div>
        </div>;
}
