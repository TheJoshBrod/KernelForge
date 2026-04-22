"""Kernel Workbench Component"""

import from react { useState, useEffect, useCallback, useRef }
import from "react-router-dom" { useNavigate, useParams }
import from .....context.ConfigContext { useConfig }
import from ...dashboard.GpuMismatchModal { GpuMismatchModal }
import from "reactflow" { useNodesState, useEdgesState }
import from .....components.ui.Icons { CodeIcon, ActivityIcon, ShieldCheckIcon }
import from .NewKernel { NewKernel }
import from ...shared.MetricsAdapter {
    to_list,
    to_num,
    normalize_op_name,
    op_matches,
    get_metric_row_for_op
}

import from .WorkbenchUtils {
    decode_param,
    canonical_op_param,
    get_layouted_elements,
    make_tree_signature
}
import from .WorkbenchEditor { WorkbenchEditor }
import from .WorkbenchLog { WorkbenchLog }
import from .WorkbenchControls { WorkbenchControls }
import from .WorkbenchTreeView { WorkbenchTreeView }
import from .WorkbenchListView { WorkbenchListView }
import from .WorkbenchNodeDetails { WorkbenchNodeDetails }

def:pub OperatorWorkbenchView(props: any) -> Any {
    projectName = props.projectName;
    opName = props.opName;

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

    left_panel_mode_state = useState("editor");  # "editor", "log", or "details"
    leftPanelMode = left_panel_mode_state[0];
    setLeftPanelMode = left_panel_mode_state[1];

    live_updates_state = useState(False);
    liveUpdates = live_updates_state[0];
    setLiveUpdates = live_updates_state[1];

    torch_op_ms_state = useState(None);
    torchOpMs = torch_op_ms_state[0];
    setTorchOpMs = torch_op_ms_state[1];

    tree_signature_ref = useRef("");
    selected_node_ref = useRef(None);
    prev_gen_tasks_len_ref = useRef(0);

    # Resizable split panels
    panel_width_state = useState(80.0);
    panelWidth = panel_width_state[0];
    setPanelWidth = panel_width_state[1];
    is_dragging_ref = useRef(False);
    container_ref = useRef(None);

    is_optimizing_state = useState(False);
    isOptimizing = is_optimizing_state[0];
    setIsOptimizing = is_optimizing_state[1];

    active_gen_tasks_state = useState([]);
    activeGenTasks = active_gen_tasks_state[0];
    setActiveGenTasks = active_gen_tasks_state[1];

    config_ctx = useConfig();
    config = config_ctx.config;

    gpu_mismatch_state = useState(False);
    gpuMismatch = gpu_mismatch_state[0];
    setGpuMismatch = gpu_mismatch_state[1];

    benchmarked_gpu_name_state = useState("");
    benchmarkedGpuName = benchmarked_gpu_name_state[0];
    setBenchmarkedGpuName = benchmarked_gpu_name_state[1];

    gpu_mismatch_modal_open_state = useState(False);
    gpuMismatchModalOpen = gpu_mismatch_modal_open_state[0];
    setGpuMismatchModalOpen = gpu_mismatch_modal_open_state[1];

    gpu_clone_busy_state = useState(False);
    gpuCloneBusy = gpu_clone_busy_state[0];
    setGpuCloneBusy = gpu_clone_busy_state[1];

    gpu_info_ref = useRef(None);

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
                metric_row = get_metric_row_for_op(metrics_data, op_for_query);
                pytorch_raw = (
                    metric_row["pytorch_ms"]
                    if metric_row and "pytorch_ms" in metric_row
                    else (
                        metric_row["pytorch"]
                        if metric_row and "pytorch" in metric_row
                        else (
                            metric_row["baseline"]
                            if metric_row and "baseline" in metric_row
                            else None
                        )
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
                    metric_row = get_metric_row_for_op(chart_data, op_for_query);
                    pytorch_raw = (
                        metric_row["pytorch_ms"]
                        if metric_row and "pytorch_ms" in metric_row
                        else (
                            metric_row["pytorch"]
                            if metric_row and "pytorch" in metric_row
                            else (
                                metric_row["baseline"]
                                if metric_row and "baseline" in metric_row
                                else None
                            )
                        )
                    );
                    torch_ms = to_num(pytorch_raw);
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
            best_node_id = (
                f"{tree_data['best_node_id']}"
                if tree_data and "best_node_id" in tree_data
                and tree_data["best_node_id"] is not None
                else None
            );

            for n in m_nodes {
                r_map[f"{n['id']}"] = n;
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

    # When a gen task finishes (activeGenTasks shrinks), immediately re-fetch the tree
    useEffect(
        lambda -> None {
            cur_len = activeGenTasks.length;
            prev_len = prev_gen_tasks_len_ref.current;
            prev_gen_tasks_len_ref.current = cur_len;
            if prev_len > 0 and cur_len < prev_len {
                fetchData(False);
            }
        },
        [activeGenTasks]
    );

    # Inject CSS keyframe animation for the pending node blink (once on mount)
    useEffect(
        lambda -> Any {
            style = document.createElement("style");
            style.id = "kforge-pending-anim";
            style.textContent = "@keyframes kf-pending { 0%, 100% { opacity: 0.85; } 50% { opacity: 0.08; } } .pending-kernel-node { animation: kf-pending 1.4s ease-in-out infinite; }";
            document.head.appendChild(style);
            return lambda -> None {
                el = document.getElementById("kforge-pending-anim");
                if el { el.remove(); }
            };
        },
        []
    );

    useEffect(
        lambda -> None { gpu_info_ref.current = config["selected_gpu_info"] if config and "selected_gpu_info" in config else None; },
        [config]
    );

    # Poll optimize job status every 3 s to drive the pending-node overlay
    useEffect(
        lambda -> Any {
            async def pollStatus -> None {
                try {
                    resp = await __jacSpawn(
                        "GetProjectStatus", "",
                        {"projectName": decode_param(resolvedProjectName), "currentGpuInfo": gpu_info_ref.current}
                    );
                    data = resp.reports[0] if resp and resp.reports else {};
                    queue_data = data["queue"] if data and "queue" in data else {};
                    active_tasks_data = queue_data["active_tasks"] if "active_tasks" in queue_data else {};
                    active_tasks_list = [];
                    cur_op = canonical_op_param(resolvedOpName);
                    for tk in Object.keys(active_tasks_data) {
                        t = active_tasks_data[tk];
                        is_gen_or_opt = tk.startsWith("gen_") or (t and "tag" in t and (t["tag"] == "[OPT]" or t["tag"] == "[GEN]")) or tk == "seq_opt" or (t and "parent_ref" in t);
                        task_op = t["op_name"] if t and "op_name" in t and t["op_name"] else "";
                        op_matches = not task_op or task_op == cur_op;
                        if is_gen_or_opt and op_matches {
                            step = t["current_step"] if "current_step" in t else "";
                            status = t["status"] if "status" in t else "";
                            if step and step != "Done" and status != "Done" {
                                att_cur = t["attempt_current"] if "attempt_current" in t else None;
                                att_max = t["attempt_max"] if "attempt_max" in t else None;
                                pref = str(t["parent_ref"]) if "parent_ref" in t and t["parent_ref"] else "";
                                info = {
                                    "id": tk,
                                    "step": step,
                                    "status": status,
                                    "attempt_current": att_cur,
                                    "attempt_max": att_max,
                                    "parent_ref": pref
                                };
                                active_tasks_list.push(info);
                            }
                        }
                    }
                    setActiveGenTasks(active_tasks_list);
                    setIsOptimizing(active_tasks_list.length > 0);
                    mismatch = data["gpu_mismatch"] if data and "gpu_mismatch" in data else False;
                    setGpuMismatch(mismatch);
                    if mismatch and "benchmarked_gpu_name" in data {
                        setBenchmarkedGpuName(str(data["benchmarked_gpu_name"]));
                    }
                } except Exception {
                    setIsOptimizing(False);
                    setActiveGenTasks([]);
                }
            }
            pollStatus();
            timer = setInterval(lambda -> None { pollStatus();}, 3000);
            return lambda -> None { clearInterval(timer);};
        },
        [resolvedProjectName]
    );

    onNodeClick = useCallback(
        lambda event: any , nd: any  -> None { setSelectedNodeId(nd.id);}, []
    );

    useEffect(
        lambda -> None {
            setNodes(
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
            );
            setEdges(
                lambda eds: list -> list { new_edges = [];for e in eds {
                    is_connected = bool(selectedNodeId) and (e.source == selectedNodeId or e.target == selectedNodeId);
                    new_edges.push({
                        "id": e.id,
                        "source": e.source,
                        "target": e.target,
                        "type": e.type,
                        "animated": is_connected,
                        "zIndex": 1 if is_connected else 0,
                        "style": {"stroke": "#ffa35b", "strokeWidth": 2} if is_connected else {"stroke": "#a4543e"}
                    });
                }return new_edges; }
            );
        },
        [selectedNodeId, setNodes, setEdges]
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

    if loading {
        return
            <div
                className="flex items-center justify-center h-screen bg-zinc-950 text-zinc-500"
            >
                Loading Operator Workbench...
            </div>;
    }

    hasTree = nodes.length > 0;

    # Build display arrays — real nodes/edges plus a blinking pending node when optimizing
    displayNodes = [];
    for n in nodes { displayNodes.push(n); }
    displayEdges = [];
    for e in edges { displayEdges.push(e); }

    pending_style = {
        "background": "#0f0e1a",
        "color": "#818cf8",
        "border": "1px dashed #6366f1",
        "borderRadius": "8px",
        "padding": "10px",
        "fontSize": "11px",
        "fontFamily": "monospace",
        "width": 100,
        "textAlign": "center"
    };

    if isOptimizing and hasTree {
        # Build map of parent_id -> rightmost real-child x so pending nodes
        # are placed to the right of existing children instead of overlapping.
        node_pos_map = {};
        for n in nodes { node_pos_map[str(n.id)] = n.position.x; }
        parent_child_max_x = {};
        for e in edges {
            src = str(e.source);
            tgt = str(e.target);
            if tgt in node_pos_map {
                tx = node_pos_map[tgt];
                if src in parent_child_max_x {
                    if tx > parent_child_max_x[src] {
                        parent_child_max_x[src] = tx;
                    }
                } else {
                    parent_child_max_x[src] = tx;
                }
            }
        }

        parent_pending_counts = {};
        for tinfo in activeGenTasks {
            pending_parent_id = None;
            pending_parent_x = 0;
            pending_parent_y = 0;
            pref = tinfo["parent_ref"] if tinfo and "parent_ref" in tinfo else "";
            parent_node_id = str(pref).replace("kernel_", "") if pref else "";
            for n in nodes {
                if str(n.id) == parent_node_id {
                    pending_parent_id = n.id;
                    pending_parent_x = n.position.x;
                    pending_parent_y = n.position.y;
                }
            }
            if pending_parent_id {
                child_y = pending_parent_y + 170;
                key = str(pending_parent_id);
                if not (key in parent_pending_counts) {
                    parent_pending_counts[key] = 0;
                }

                offset_idx = parent_pending_counts[key];
                parent_pending_counts[key] = offset_idx + 1;

                task_id = tinfo["id"] if "id" in tinfo else str(offset_idx);
                node_id = f"pending-node-{task_id}";

                # Place pending node after the rightmost real child, or below
                # the parent if no real children exist yet.
                base_x = (parent_child_max_x[key] + 140) if key in parent_child_max_x else pending_parent_x;
                pending_x = base_x + offset_idx * 140;

                step_label = tinfo["step"] if "step" in tinfo and tinfo["step"] else "Working...";
                displayNodes.push({
                    "id": node_id,
                    "position": {"x": pending_x, "y": child_y},
                    "data": {"label": step_label},
                    "className": "pending-kernel-node",
                    "style": pending_style,
                    "type": "default"
                });
                displayEdges.push({
                    "id": f"e-pending-{task_id}",
                    "source": pending_parent_id,
                    "target": node_id,
                    "type": "smoothstep",
                    "animated": True,
                    "style": {"stroke": "#6366f1", "strokeDasharray": "5,5"}
                });
            }
        }
    } elif isOptimizing {
        # No tree yet — standalone pending root node(s)
        idx = 0;
        for tinfo in activeGenTasks {
            task_id = tinfo["id"] if "id" in tinfo else str(idx);
            step_label = tinfo["step"] if "step" in tinfo and tinfo["step"] else "Working...";
            displayNodes.push({
                "id": f"pending-node-{task_id}",
                "position": {"x": idx * 200, "y": 0},
                "data": {"label": step_label},
                "className": "pending-kernel-node",
                "style": pending_style,
                "type": "default"
            });
            idx += 1;
        }
    }

    resolvedProjectNameClean = decode_param(resolvedProjectName);
    resolvedOpNameClean = canonical_op_param(resolvedOpName);

    gpu_name = "";
    if config and "selected_gpu_info" in config and config["selected_gpu_info"] {
        _sgi = config["selected_gpu_info"];
        gpu_name = _sgi["name"] if "name" in _sgi and _sgi["name"] else "";
    }

    async def handleCloneForGpu -> None {
        setGpuCloneBusy(True);
        try {
            resp = await __jacSpawn("CloneProjectForGpu", "", {
                "sourceProjectName": decode_param(resolvedProjectName),
                "currentGpuInfo": config["selected_gpu_info"] if config and "selected_gpu_info" in config else None
            });
            data = resp.reports[0] if resp and resp.reports else {};
            if data and "success" in data and data["success"] {
                navigate(f"/project/{data['newProjectName']}");
            }
        } except Exception {

        } finally {
            setGpuCloneBusy(False);
            setGpuMismatchModalOpen(False);
        }
    }

    left_content = [];
    if leftPanelMode == "editor" {
        left_content.push(
            <WorkbenchEditor
                selectedNodeId={selectedNodeId}
                selectedNodeData={selectedNodeData}
            />
        );
    } elif leftPanelMode == "log" {
        left_content.push(
            <WorkbenchLog
                selectedNodeId={selectedNodeId}
                rawNodes={rawNodes}
                nodes={nodes}
                edges={edges}
            />
        );
    } else {
        left_content.push(
            <WorkbenchNodeDetails
                selectedNodeId={selectedNodeId}
                selectedNodeData={selectedNodeData}
            />
        );
    }

    def handleDividerMouseDown(e: any) -> None {
        e.preventDefault();
        is_dragging_ref.current = True;
    }

    def handleMouseMove(e: any) -> None {
        if is_dragging_ref.current {
            container = container_ref.current;
            if container {
                rect = container.getBoundingClientRect();
                new_width = ((e.clientX - rect.left) / rect.width) * 100;
                if new_width < 20 { new_width = 20; }
                if new_width > 80 { new_width = 80; }
                setPanelWidth(new_width);
            }
        }
    }

    def handleMouseUp(e: any) -> None {
        is_dragging_ref.current = False;
    }

    return
        <>
        <div
            ref={container_ref}
            className="flex h-screen w-full bg-zinc-950 text-zinc-100 overflow-hidden"
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
        >
            <div style={{"width": panelWidth + "%"}} className="flex flex-col min-w-0">
                <div
                    className="h-10 bg-zinc-900/50 border-b border-zinc-800 flex items-center px-4 gap-1 flex-shrink-0"
                >
                    <button
                        className={"px-3 py-1.5 text-xs font-bold rounded transition-all " + (
                            "bg-zinc-700 text-white shadow-sm"
                            if leftPanelMode == "editor"
                            else "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50"
                        )}
                        onClick={lambda : setLeftPanelMode("editor")}
                    >
                        Editor
                    </button>
                    <button
                        className={"px-3 py-1.5 text-xs font-bold rounded transition-all " + (
                            "bg-zinc-700 text-white shadow-sm"
                            if leftPanelMode == "log"
                            else "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50"
                        )}
                        onClick={lambda : setLeftPanelMode("log")}
                    >
                        Log
                    </button>
                    <button
                        className={"px-3 py-1.5 text-xs font-bold rounded transition-all " + (
                            "bg-zinc-700 text-white shadow-sm"
                            if leftPanelMode == "details"
                            else "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50"
                        )}
                        onClick={lambda : setLeftPanelMode("details")}
                    >
                        Details
                    </button>
                </div>
                {left_content}
            </div>
            <div
                className="w-1 bg-zinc-800 hover:bg-indigo-500/50 cursor-col-resize flex-shrink-0 transition-colors select-none"
                onMouseDown={handleDividerMouseDown}
            />
            <div style={{"width": (100 - panelWidth) + "%"}} className="flex flex-col bg-zinc-950 min-w-0">
                <div
                    className="flex-1 border-b border-zinc-800 relative bg-[radial-gradient(circle_at_18%_14%,rgba(225,112,85,0.12),transparent_42%),radial-gradient(circle_at_84%_78%,rgba(20,184,166,0.10),transparent_45%),#09090b]"
                >
                    <WorkbenchControls
                        viewMode={viewMode}
                        onViewModeChange={setViewMode}
                        liveUpdates={liveUpdates}
                        onLiveUpdatesChange={setLiveUpdates}
                        fetchData={fetchData}
                    />
                    {(
                        <WorkbenchTreeView
                            displayNodes={displayNodes}
                            displayEdges={displayEdges}
                            onNodesChange={onNodesChange}
                            onEdgesChange={onEdgesChange}
                            onNodeClick={onNodeClick}
                            hasTree={hasTree}
                            isOptimizing={isOptimizing}
                        />
                    ) if viewMode == "graph" else (
                        <WorkbenchListView
                            viewMode={viewMode}
                            sortedNodes={sortedNodes}
                            selectedNodeId={selectedNodeId}
                            onNodeSelect={setSelectedNodeId}
                            torchDisplayMs={torchOpMs}
                        />
                    )}
                </div>
                <div className="p-4 border-t border-zinc-800 bg-zinc-900/30">
                    <NewKernel
                        projectName={resolvedProjectNameClean}
                        opName={resolvedOpNameClean}
                        hasTree={hasTree}
                        gpuMismatch={gpuMismatch}
                        onGpuMismatch={lambda -> None { setGpuMismatchModalOpen(True);}}
                        onUploadSuccess={lambda -> None {}}
                    />
                </div>
            </div>
        </div>
        <GpuMismatchModal
            isOpen={gpuMismatchModalOpen}
            onClose={lambda -> None { setGpuMismatchModalOpen(False);}}
            benchmarkedGpuName={benchmarkedGpuName}
            currentGpuName={gpu_name}
            onClone={handleCloneForGpu}
            onReturnToProject={lambda -> None { navigate(f"/project/{decode_param(resolvedProjectName)}");}}
            cloneBusy={gpuCloneBusy}
        />
        </>;
}
