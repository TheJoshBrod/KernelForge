"""Kernel Workbench Component"""

import from react { useState, useEffect, useCallback, useRef }
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
import from ....components.editor.ForgePythonEditor { ForgePythonEditor }

def:pub OperatorWorkbenchView -> Any {
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

    log_collapsed_state = useState(False);
    logCollapsed = log_collapsed_state[0];
    setLogCollapsed = log_collapsed_state[1];

    live_updates_state = useState(False);
    liveUpdates = live_updates_state[0];
    setLiveUpdates = live_updates_state[1];

    torch_op_ms_state = useState(None);
    torchOpMs = torch_op_ms_state[0];
    setTorchOpMs = torch_op_ms_state[1];

    tree_signature_ref = useRef("");
    selected_node_ref = useRef(None);

    # --- Helpers ---
    def to_list(items: Any) -> list {
        if not items {
            return [];
        }
        try {
            if Array.isArray(items) {
                return items;
            }
        } except Exception {
        # no-op
        }
        normalized = [];
        try {
            for item in items {
                normalized.push(item);
            }
            return normalized;
        } except Exception {
        # no-op
        }
        try {
            return Object.values(items);
        } except Exception {
            return [];
        }
    }

    def to_num(value: Any) -> Any {
        if value is None {
            return None;
        }
        try {
            text = f"{value}".trim();
            if not text {
                return None;
            }
            parsed = parseFloat(text);
            if parsed == parsed {
                return parsed;
            }
        } except Exception {
        # no-op
        }
        return None;
    }

    def normalize_op_name(value: Any) -> str {
        text = str(value or "").trim().toLowerCase();
        if not text {
            return "";
        }
        prefix = "torch.nn.functional.";
        if text.startsWith(prefix) {
            text = "torch_nn_functional_" + text.replace(prefix, "");
        }
        return text.replace(".", "_").replace("/", "_");
    }

    def decode_param(value: Any) -> str {
        text = str(value or "");
        if not text {
            return "";
        }
        try {
            return decodeURIComponent(text);
        } except Exception {
            return text;
        }
    }

    def canonical_op_param(value: Any) -> str {
        text = decode_param(value).trim();
        if not text {
            return "";
        }
        lowered = text.toLowerCase();
        marker = "/kernels/";
        idx = lowered.indexOf(marker);
        if idx >= 0 {
            text = text.slice(0, idx);
        }
        marker = "\\kernels\\";
        idx = lowered.indexOf(marker);
        if idx >= 0 {
            text = text.slice(0, idx);
        }
        return text.trim();
    }

    def op_matches(candidate: Any, target: str) -> bool {
        cand = normalize_op_name(candidate);
        if not cand or not target {
            return False;
        }
        if cand == target {
            return True;
        }
        if target.startsWith(cand + "_") {
            return True;
        }
        if cand.startsWith(target + "_") {
            return True;
        }
        return False;
    }

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
            # Tree data can be incomplete while generation/optimization is running.
            # If an edge references a node that doesn't exist yet, skip it to avoid UI crashes.
            if not (pid in children_map) {
                continue;
            }
            if not (cid in node_map) {
                continue;
            }
            children_map[pid].push(cid);
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

            raw_node = node_map[cid] if cid in node_map else None;
            if not raw_node {
                continue;
            }
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
            if not (e["parent_id"] in node_map) {
                continue;
            }
            if not (e["child_id"] in node_map) {
                continue;
            }
            rf_edges.push(
                {
                    "id": f"e{e['parent_id']}-{e['child_id']}",
                    "source": f"{e['parent_id']}",
                    "target": f"{e['child_id']}",
                    "type": "smoothstep",
                    "animated": True,
                    "style": {"stroke": "#e17055"}
                }
            );
        }

        return {"nodes": layouted_nodes, "edges": rf_edges};
    }

    def make_tree_signature(nodes_data: list, edges_data: list) -> str {
        node_parts = [];
        for n in nodes_data {
            if not n {
                continue;
            }
            nid = str(n["id"]) if "id" in n else "";
            nval = Number(n["value"]) if "value" in n else 0;
            nts = str(n["timestamp"]) if "timestamp" in n and n["timestamp"] else "";
            ncode = str(n["code"]) if "code" in n and n["code"] else "";
            node_parts.push(f"{nid}:{nval}:{nts}:{ncode.length}");
        }
        node_parts.sort();

        edge_parts = [];
        for e in edges_data {
            if not e {
                continue;
            }
            pid = str(e["parent_id"]) if "parent_id" in e else "";
            cid = str(e["child_id"]) if "child_id" in e else "";
            edge_parts.push(f"{pid}->{cid}");
        }
        edge_parts.sort();

        return node_parts.join("|") + "||" + edge_parts.join("|");
    }

    async def fetchData(showLoader: bool = False) -> None {
        project_for_query = decode_param(resolvedProjectName);
        op_for_query = canonical_op_param(resolvedOpName);
        if not project_for_query or not op_for_query {
            if showLoader {
                setLoading(False);
            }
            return;
        }
        if showLoader {
            setLoading(True);
        }
        try {
            metrics_req = __jacSpawn(
                "GetProjectMetrics", "", {"projectName": project_for_query}
            );
            tree_req = __jacSpawn(
                "GetMctsTree",
                "",
                {"projectName": project_for_query, "opName": op_for_query}
            );
            torch_ms = None;
            try {
                metrics_resp = await metrics_req;
                metrics_data = metrics_resp.reports[0]
                if metrics_resp and metrics_resp.reports
                else {};
                target_op = normalize_op_name(op_for_query);
                rows_by_op = (
                    metrics_data["rows_by_op"]
                    if metrics_data and "rows_by_op" in metrics_data and metrics_data["rows_by_op"]
                    else {}
                );
                metric_row = rows_by_op[target_op]
                if rows_by_op and target_op in rows_by_op
                else None;
                if not metric_row and rows_by_op {
                    for row_key in Object.keys(rows_by_op) {
                        if op_matches(row_key, target_op) {
                            metric_row = rows_by_op[row_key];
                            break;
                        }
                    }
                }
                if not metric_row {
                    metric_rows = to_list(
                        metrics_data["rows"]
                        if metrics_data and "rows" in metrics_data
                        else []
                    );
                    for row in metric_rows {
                        if not row {
                            continue;
                        }
                        row_name = (
                            row["name"]
                            if "name" in row
                            else (row["op"] if "op" in row else "")
                        );
                        if op_matches(row_name, target_op) {
                            metric_row = row;
                            break;
                        }
                    }
                }
                pytorch_raw = (
                    metric_row["pytorch_ms"]
                    if metric_row and "pytorch_ms" in metric_row
                    else (
                        metric_row["pytorch"]
                        if metric_row and "pytorch" in metric_row
                        else None
                    )
                );
                torch_ms = to_num(pytorch_raw);
            } except Exception as metrics_err {
                console.warn("Failed to load project metrics:", metrics_err);
            }
            if torch_ms is None {
                try {
                    chart_resp = await __jacSpawn(
                        "GetDashboardCharts", "", {"projectName": project_for_query}
                    );
                    chart_data = chart_resp.reports[0]
                    if chart_resp and chart_resp.reports
                    else {};
                    target_op = normalize_op_name(op_for_query);
                    speed_rows = to_list(
                        chart_data["speed_comparison"]
                        if chart_data and "speed_comparison" in chart_data
                        else []
                    );
                    for row in speed_rows {
                        if not row {
                            continue;
                        }
                        row_name = (
                            row["name"]
                            if "name" in row
                            else (row["op"] if "op" in row else "")
                        );
                        if not op_matches(row_name, target_op) {
                            continue;
                        }
                        pytorch_raw = row["pytorch"] if "pytorch" in row else None;
                        if pytorch_raw is None and "baseline" in row {
                            pytorch_raw = row["baseline"];
                        }
                        torch_ms = to_num(pytorch_raw);
                        break;
                    }
                } except Exception as chart_err {
                    console.warn("Failed to load dashboard chart fallback:", chart_err);
                }
            }

            tree_resp = await tree_req;
            tree_data = tree_resp.reports[0] if tree_resp and tree_resp.reports else {};
            tree_pytorch = to_num(
                tree_data["pytorch_ms"]
                if tree_data and "pytorch_ms" in tree_data
                else None
            );
            if tree_pytorch and tree_pytorch > 0 {
                torch_ms = tree_pytorch;
            }
            setTorchOpMs(torch_ms);

            m_nodes = tree_data["nodes"] if tree_data and "nodes" in tree_data else [];
            m_edges = tree_data["edges"] if tree_data and "edges" in tree_data else [];
            next_signature = make_tree_signature(m_nodes, m_edges);
            if (
                (not showLoader)
                and tree_signature_ref.current
                and tree_signature_ref.current == next_signature
            ) {
                return;
            }
            tree_signature_ref.current = next_signature;

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

            current_selected = selected_node_ref.current;
            if best_node_id {
                if not current_selected or not (current_selected in r_map) {
                    setSelectedNodeId(best_node_id);
                }
            } else {
                if current_selected {
                    setSelectedNodeId(None);
                }
            }
        } except Exception as e {
            console.error("Failed to fetch operator workbench data:", e);
            setTorchOpMs(None);
        } finally {
            if showLoader {
                setLoading(False);
            }
        }
    }

    useEffect(
        lambda -> None { fetchData(True);}, [resolvedProjectName, resolvedOpName]
    );

    useEffect(
        lambda -> None { setTorchOpMs(None);tree_signature_ref.current = "";},
        [resolvedProjectName, resolvedOpName]
    );

    useEffect(
        lambda -> None { selected_node_ref.current = selectedNodeId;},
        [selectedNodeId]
    );

    useEffect(
        lambda -> Any { if not liveUpdates {
            return lambda -> None {};
        }timer = setInterval(lambda -> None { fetchData(False);}, 8000);return lambda -> None { clearInterval(
            timer
        );}; },
        [resolvedProjectName, resolvedOpName, liveUpdates]
    );

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
                    new_style["border"] = "2px solid #e17055";
                    new_style["background"] = "rgba(225, 112, 85, 0.12)";
                    new_style["boxShadow"] = "0 0 10px rgba(225, 112, 85, 0.28)";
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
        # Avoid Array.sort comparator type issues in Jac transpilation.
        i = 0;
        while i < sortedNodes.length {
            j = 0;
            while j < sortedNodes.length - 1 {
                a = sortedNodes[j];
                b = sortedNodes[j + 1];
                av = Number(a["value"]) if a and "value" in a else 0;
                bv = Number(b["value"]) if b and "value" in b else 0;
                if av > bv {
                    tmp = sortedNodes[j];
                    sortedNodes[j] = sortedNodes[j + 1];
                    sortedNodes[j + 1] = tmp;
                }
                j += 1;
            }
            i += 1;
        }
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
                        "border-orange-500/80 bg-orange-500/10 shadow-[0_0_15px_rgba(225,112,85,0.18)]"
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
                            "text-orange-300"
                            if f"{n['id']}" == selectedNodeId
                            else "text-zinc-300"
                        )}
                    >
                        Kernel_{n["id"]}:
                    </div>
                    <div
                        className="font-mono text-teal-300 text-sm font-bold flex-1"
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
                    "border-orange-500/50 bg-orange-500/10"
                    if f"{item['id']}" == selectedNodeId
                    else "border-zinc-800 bg-zinc-900/50"
                )}
                ref={lambda el: any  -> None { if f"{item['id']}" == selectedNodeId
                and el {
                    el.scrollIntoView({"behavior": "smooth", "block": "nearest"});
                }}}
            >
                <div className="flex justify-between mb-1">
                    <span className="font-mono font-bold text-orange-300">
                        kernel_{item["id"]}
                    </span>
                    <span className="text-teal-300 font-mono">
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
                Loading Operator Workbench...
            </div>;
    }

    hasTree = nodes.length > 0;
    torch_display_ms = to_num(torchOpMs);
    torch_display_text = "--";
    if torch_display_ms and torch_display_ms > 0 {
        torch_display_text = f"{Number(torch_display_ms).toFixed(9)} ms";
    }
    resolvedProjectNameClean = decode_param(resolvedProjectName);
    resolvedOpNameClean = canonical_op_param(resolvedOpName);

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
                            <CodeIcon className="w-4 h-4"/>
                            Kernel Code
                        </span>
                        <span className="text-xs font-mono text-orange-300">
                            Node{selectedNodeId or "?"}
                        </span>
                    </div>
                    <div className="flex-1 min-h-0 relative bg-[#0d0a08] border-l-2 border-[var(--accent)]/30">
                        <ForgePythonEditor
                            value={selectedNodeData["code"] if "code" in selectedNodeData else ""}
                            readOnly={True}
                            language="cpp"
                            placeholder="// Select a node to view code"
                        />
                    </div>
                </div>
                <div
                    className={"flex flex-col bg-zinc-900/20 " + (
                        "h-10" if logCollapsed else "h-[40%]"
                    )}
                >
                    <div
                        className={"h-10 bg-zinc-900/50 flex items-center px-4 justify-between " + (
                            "border-b border-zinc-800" if not logCollapsed else ""
                        )}
                    >
                        <span
                            className="text-xs font-bold text-zinc-500 uppercase flex items-center gap-2"
                        >
                            <ActivityIcon className="w-4 h-4"/>
                            Iteration Log
                        </span>
                        <button
                            className="text-[11px] text-zinc-400 hover:text-zinc-200 px-2 py-1 rounded border border-zinc-700 bg-zinc-900/50"
                            onClick={lambda -> None { setLogCollapsed(not logCollapsed);}}
                        >
                            {"Expand" if logCollapsed else "Collapse"}
                        </button>
                    </div>
                    {<div
                        className="flex-1 overflow-y-auto p-4 space-y-2 flex flex-col-reverse"
                    >
                        {logViewItems}
                    </div>
                    if not logCollapsed
                    else <span></span>}
                </div>
            </div>
            <div className="w-1/2 flex flex-col bg-zinc-950">
                <div
                    className="h-[50%] border-b border-zinc-800 relative bg-[radial-gradient(circle_at_18%_14%,rgba(225,112,85,0.12),transparent_42%),radial-gradient(circle_at_84%_78%,rgba(20,184,166,0.10),transparent_45%),#09090b]"
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
                        <button
                            className="px-3 py-1.5 text-xs font-bold rounded transition-all text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50"
                            onClick={lambda -> None { fetchData(False);}}
                        >
                            Refresh
                        </button>
                        <button
                            className={"px-3 py-1.5 text-xs font-bold rounded transition-all " + (
                                "bg-teal-700/80 text-white shadow-sm"
                                if liveUpdates
                                else "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50"
                            )}
                            onClick={lambda -> None { setLiveUpdates(not liveUpdates);}}
                        >
                            {"Live On" if liveUpdates else "Live Off"}
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
                        <Background
                            id="compute-grid"
                            variant="lines"
                            color="rgba(20, 184, 166, 0.18)"
                            gap={32}
                        />
                        <Background
                            id="compute-dust"
                            variant="dots"
                            color="rgba(225, 112, 85, 0.16)"
                            gap={22}
                            size={1}
                        />
                    </ReactFlow>
                    if viewMode == "graph" and hasTree
                    else <div
                        className="h-full w-full flex items-center justify-center p-6 text-center"
                    >
                        <div className="max-w-sm">
                            <div className="text-sm font-semibold text-zinc-200">
                                No kernels yet for this operator
                            </div>
                            <div className="mt-1 text-xs text-zinc-500">
                                Generate a new root node to start building an optimization tree.
                            </div>
                        </div>
                    </div>
                    if viewMode == "graph"
                    else <div className="h-full overflow-y-auto p-4 pt-14 space-y-2">
                        {listViewItems}{<div
                            className="text-center py-10 text-zinc-600 italic text-sm"
                        >
                            No kernels found.
                        </div>
                        if sortedNodes.length == 0
                        else <span/>}
                    </div>}
                </div>
                <div className="flex-1 p-6 overflow-y-auto">
                    <div
                        className="flex justify-between items-center mb-4 border-b border-zinc-800 pb-2"
                    >
                        <h3
                            className="text-lg font-bold text-white flex items-center gap-2"
                        >
                            <ShieldCheckIcon className="w-5 h-5 text-teal-400"/>
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
                            <span className="font-mono text-orange-300">
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
                            <span className="font-mono text-teal-300">
                                {Number(selectedNodeData["value"] or 0).toFixed(6)}
                            </span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-zinc-500 text-sm">
                                Best Subtree
                            </span>
                            <span className="font-mono text-teal-300">
                                {Number(selectedNodeData["best_subtree_value"] or 0).toFixed(
                                    6
                                )}
                            </span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-zinc-500 text-sm">
                                PyTorch Op Time
                            </span>
                            <span className="font-mono text-teal-300">
                                {torch_display_text}
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
                        projectName={resolvedProjectNameClean}
                        opName={resolvedOpNameClean}
                        hasTree={hasTree}
                    />
                </div>
            </div>
        </div>;
}
