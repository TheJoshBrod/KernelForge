"""Data Flow View — Operator compute graph with kernel fusion management."""

import from react { useState, useEffect, useRef, useMemo }
import from "react-router-dom" { useParams, useNavigate }
import from reactflow {
    ReactFlow,
    Background,
    Controls,
    MiniMap,
    useNodesState,
    useEdgesState
}
import from "./dagreWrap.js" { createGraph, layoutGraph as dagreLayout }
import from "./fusionColors.js" { FUSION_COLORS }
import from ....components.ui.GlowButton { GlowButton }
import from ....components.ui.Icons { BrainIcon, RefreshIcon, DownloadIcon, MenuIcon }
import from ....context.ConfigContext { useConfig }
import from .OpNode { OpNode }
import from .FusionGroupNode { FusionGroupNode }
import from .FusionPanel { FusionPanel }
import from .FusionProposalModal { FusionProposalModal }

def:pub DataFlowView -> Any {
    params = useParams();
    projectName = params.projectName;
    navigate = useNavigate();
    config_ctx = useConfig();
    config = config_ctx.config;

    nodes_state = useNodesState([]);
    nodes = nodes_state[0];
    setNodes = nodes_state[1];
    onNodesChange = nodes_state[2];

    edges_state = useEdgesState([]);
    edges = edges_state[0];
    setEdges = edges_state[1];
    onEdgesChange = edges_state[2];

    profile_status_state = useState("idle");
    profileStatus = profile_status_state[0];
    setProfileStatus = profile_status_state[1];

    fusion_groups_state = useState([]);
    fusionGroups = fusion_groups_state[0];
    setFusionGroups = fusion_groups_state[1];

    all_nodes_state = useState([]);
    allNodes = all_nodes_state[0];
    setAllNodes = all_nodes_state[1];

    selected_pattern_state = useState(None);
    selectedPatternName = selected_pattern_state[0];
    setSelectedPatternName = selected_pattern_state[1];

    panel_open_state = useState(False);
    panelOpen = panel_open_state[0];
    setPanelOpen = panel_open_state[1];

    proposal_modal_state = useState(False);
    proposalModalOpen = proposal_modal_state[0];
    setProposalModalOpen = proposal_modal_state[1];

    pollRef = useRef(None);

    # Stable nodeTypes reference — prevents ReactFlow re-mount on each render
    nodeTypes_memo = useMemo(
        lambda -> any { return {"opNode": OpNode, "fusionGroup": FusionGroupNode}; },
        []
    );

    def buildLayout(enriched_data: dict, fg_list: list) -> None {
        if not enriched_data or not ("nodes" in enriched_data) {
            return;
        }

        enriched_nodes = enriched_data["nodes"];
        enriched_edges = enriched_data["edges"] if "edges" in enriched_data else [];

        g = createGraph();
        g.setGraph({"rankdir": "TB", "ranksep": 120, "nodesep": 120, "edgesep": 40});
        g.setDefaultEdgeLabel(lambda -> dict { return {}; });

        # Build immediate adjacency for topological partitioning
        local_adj = {};
        for e in enriched_edges {
            s = e["source"] if "source" in e else "";
            t = e["target"] if "target" in e else "";
            if s and t {
                if s not in local_adj {
                    local_adj[s] = [];
                }
                local_adj[s].push(t);
                if t not in local_adj {
                    local_adj[t] = [];
                }
                local_adj[t].push(s);
            }
        }

        partitioned_fg_list = [];
        for fg in fg_list {
            orig_members = fg["members"] if "members" in fg else [];
            if not orig_members {
                partitioned_fg_list.push(fg);
                continue;
            }

            mem_set = {};
            for m in orig_members {
                mem_set[m] = True;
            }
            visited = {};

            inst_count = 0;
            for start_node in orig_members {
                if start_node not in visited {
                    comp = [];
                    queue = [start_node];
                    visited[start_node] = True;

                    while queue.length > 0 {
                        curr = queue.pop();
                        comp.push(curr);
                        nbrs = local_adj[curr] if curr in local_adj else [];
                        for nbr in nbrs {
                            if nbr in mem_set and nbr not in visited {
                                visited[nbr] = True;
                                queue.push(nbr);
                            }
                        }
                    }

                    # Create a simple shallow copy to avoid .keys() on JS objects
                    new_fg = {
                        "id": fg["id"] if "id" in fg else "",
                        "name": fg["name"] if "name" in fg else "",
                        "pattern_name": fg["pattern_name"]
                            if "pattern_name" in fg
                            else "",
                        "status": fg["status"] if "status" in fg else "",
                        "estimated_speedup": fg["estimated_speedup"]
                            if "estimated_speedup" in fg
                            else None,
                        "members": comp
                    };

                    if orig_members.length > comp.length {
                        new_fg["id"] = str(new_fg["id"]) + "_inst_" + str(inst_count);
                    }
                    partitioned_fg_list.push(new_fg);
                    inst_count = inst_count + 1;
                }
            }
        }

        # active_fg_list preserves insertion order — JS for-in on plain objects is unreliable.
        member_to_group = {};
        active_fg_ids = {};
        active_fg_list = [];
        pattern_color_map = {};
        color_counter = 0;
        for fg in partitioned_fg_list {
            fg_status = fg["status"] if "status" in fg else "";
            if fg_status == "rejected" {
                continue;
            }
            fg_id = fg["id"] if "id" in fg else "";
            if not fg_id {
                continue;
            }
            members = fg["members"] if "members" in fg else [];
            fg_pattern = fg["pattern_name"] if "pattern_name" in fg else fg_id;
            if fg_pattern not in pattern_color_map {
                pattern_color_map[fg_pattern] = color_counter;
                color_counter = color_counter + 1;
            }
            cidx_raw = pattern_color_map[fg_pattern];
            fg_name = fg["name"] if "name" in fg else fg_id;
            fg_speedup = fg["estimated_speedup"] if "estimated_speedup" in fg else None;
            fg_status_val = fg_status;

            # Register group node with dagre (dagre computes size from children)
            g.setNode(
                fg_id,
                {
                    "label": fg_name,
                    "clusterLabelPos": "top",
                    "paddingTop": 60,
                    "paddingBottom": 24,
                    "paddingLeft": 24,
                    "paddingRight": 24
                }
            );
            active_fg_ids[fg_id] = {
                "id": fg_id,
                "color_index": cidx_raw,
                "name": fg_name,
                "speedup": fg_speedup,
                "status": fg_status_val
            };
            active_fg_list.push(fg_id);

            for mem_id in members {
                member_to_group[mem_id] = fg_id;
            }
        }

        for n in enriched_nodes {
            n_id = n["id"] if "id" in n else "";
            if not n_id {
                continue;
            }
            g.setNode(n_id, {"width": 190, "height": 80});
            fg_id = member_to_group[n_id] if n_id in member_to_group else None;
            if fg_id {
                g.setParent(n_id, fg_id);
            }
        }

        for e in enriched_edges {
            src = e["source"] if "source" in e else "";
            tgt = e["target"] if "target" in e else "";
            if src and tgt {
                g.setEdge(src, tgt);
            }
        }

        dagreLayout(g);

        new_nodes = [];

        # Build ReactFlow group nodes from dagre output.
        # Also cache each group's layout rect so child nodes can compute relative positions
        # without a second g.node() call per child.
        group_rects = {};
        for fg_id in active_fg_list {
            fg_info = active_fg_ids[fg_id];
            pos = g.node(fg_id);
            if not pos {
                continue;
            }
            cidx = fg_info["color_index"];
            color = FUSION_COLORS[cidx % FUSION_COLORS.length];
            group_w = pos.width if pos.width else 220;
            group_h = pos.height if pos.height else 120;
            group_left = pos.x - group_w / 2;
            group_top = pos.y - group_h / 2;
            HEADER_PAD = 36;
            group_h = group_h + HEADER_PAD;
            group_rects[fg_id] = {
                "left": group_left,
                "top": group_top,
                "pad": HEADER_PAD,
                "w": group_w,
                "h": group_h
            };

            new_nodes.push(
                {
                    "id": fg_id,
                    "type": "fusionGroup",
                    "position": {"x": group_left, "y": group_top},
                    "style": {"width": group_w, "height": group_h, "zIndex": -1},
                    "data": {
                        "name": fg_info["name"],
                        "estimated_speedup": fg_info["speedup"],
                        "color": color,
                        "status": fg_info["status"]
                    },
                    "selectable": True
                }
            );
        }

        # Build ReactFlow op nodes
        for n in enriched_nodes {
            n_id = n["id"] if "id" in n else "";
            if not n_id {
                continue;
            }
            pos = g.node(n_id);
            if not pos {
                continue;
            }

            fg_id = member_to_group[n_id] if n_id in member_to_group else None;
            node_entry = {
                "id": n_id,
                "type": "opNode",
                "data": {
                    "op": n["op"] if "op" in n else "",
                    "bench_op": n["bench_op"] if "bench_op" in n else "",
                    "pytorch_ms": n["pytorch_ms"] if "pytorch_ms" in n else None,
                    "call_count": n["call_count"] if "call_count" in n else 0,
                    "winner": n["winner"] if "winner" in n else "",
                    "speedup": n["speedup"] if "speedup" in n else None,
                    "status": n["status"] if "status" in n else ""
                }
            };

            if fg_id and fg_id in group_rects {
                rect = group_rects[fg_id];
                node_entry["parentNode"] = fg_id;
                node_entry["extent"] = "parent";
                node_entry["position"] = {
                    "x": pos.x - 90 - rect["left"],
                    "y": pos.y - 38 - rect["top"] + rect["pad"]
                };
            } else {
                node_entry["position"] = {"x": pos.x - 90, "y": pos.y - 38};
            }

            new_nodes.push(node_entry);
        }

        new_edges = [];
        idx = 0;
        for e in enriched_edges {
            src = e["source"] if "source" in e else "";
            tgt = e["target"] if "target" in e else "";
            if not src or not tgt {
                idx = idx + 1;
                continue;
            }
            if src in active_fg_ids or tgt in active_fg_ids {
                idx = idx + 1;
                continue;
            }

            src_fg = member_to_group[src] if src in member_to_group else None;
            tgt_fg = member_to_group[tgt] if tgt in member_to_group else None;

            # Use raw truthy check, Jac 'is not None' compiles to '!== undefined' and lets 'null' pass
            intra = bool(src_fg) and bool(tgt_fg) and src_fg == tgt_fg;

            edge_entry = {
                "id": f"e_{idx}",
                "source": src,
                "target": tgt,
                "type": "default"
            };

            if intra {
                fg_info = active_fg_ids[src_fg] if src_fg in active_fg_ids else {};
                cidx = fg_info["color_index"] if "color_index" in fg_info else 0;
                edge_color = FUSION_COLORS[cidx % FUSION_COLORS.length]["border"];
                edge_entry["animated"] = True;
                edge_entry["style"] = {"stroke": edge_color, "strokeWidth": 2};
            } else {
                edge_entry["style"] = {"stroke": "var(--border)", "strokeWidth": 1};
            }

            new_edges.push(edge_entry);
            idx = idx + 1;
        }

        setNodes(new_nodes);
        setEdges(new_edges);
    }

    async def refreshDAG -> None {
        try {
            if projectName {
                resp = await __jacSpawn(
                    "GetEnrichedDAG", "", {"projectName": projectName}
                );
                data = resp.reports[0]
                    if resp and resp.reports and resp.reports.length > 0
                    else {};
                if data {
                    enriched_nodes = data["nodes"] if "nodes" in data else [];
                    fg_list = data["fusion_groups"] if "fusion_groups" in data else [];
                    setAllNodes(enriched_nodes);
                    setFusionGroups(fg_list);
                    buildLayout(data, fg_list);
                    if pollRef.current {
                        clearInterval(pollRef.current);
                        pollRef.current = None;
                        setProfileStatus("idle");
                    }
                }
            }
        } except Exception as e {
            console.error("Failed to fetch DAG:", e);
        }
    }

    # Continuous auto-update polling (15s idle, 3s when profiling)
    useEffect(
        lambda -> None { refreshDAG();def tick() -> None {
            if window and window["__kf_data_flow_busy"] {
                return;
            }
            window["__kf_data_flow_busy"] = True;
            refreshDAG().finally(
                lambda -> None { window["__kf_data_flow_busy"] = False;}
            );
        } interval_ms = 3000 if profileStatus == "running" else 15000;bgPollId = setInterval(
            tick, interval_ms
        );return lambda -> None { clearInterval(bgPollId);}; },
        [projectName, profileStatus]
    );

    async def startProfile -> None {
        try {
            setProfileStatus("running");
            await __jacSpawn(
                "StartProfile", "", {"projectName": projectName, "force": True}
            );
            # Auto-polling will handle the refresh due to profileStatus dependency
            setTimeout(lambda -> None { setProfileStatus("idle");}, 120000);
        } except Exception as e {
            console.error("Failed to start profile:", e);
            setProfileStatus("idle");
        }
    }

    def exportToMermaid -> None {
        node_labels = {};
        for n in nodes {
            n_type = n["type"] if "type" in n else "";
            if n_type == "opNode" {
                n_data = n["data"] if "data" in n else {};
                op = n_data["op"] if "op" in n_data else n["id"];
                node_labels[n["id"]] = op;
            }
        }
        mermaid = "graph TD;\n";
        for edge in edges {
            source_id = edge["source"];
            target_id = edge["target"];
            s_label = node_labels[source_id] if source_id in node_labels else source_id;
            t_label = node_labels[target_id] if target_id in node_labels else target_id;
            mermaid = mermaid + f"    {source_id}[\"{s_label}\"] --> {target_id}[\"{t_label}\"];\n";
        }
        navigator.clipboard.writeText(mermaid);
        alert("Mermaid graph copied to clipboard!");
    }

    def onNodeClick(event: any, rf_node: any) -> None {
        n_type = rf_node["type"] if rf_node and "type" in rf_node else "";
        if n_type == "fusionGroup" {
            fg_id = rf_node["id"] if "id" in rf_node else "";
            # Find the pattern_name for this group id
            for fg in fusionGroups {
                if fg["id"] == fg_id {
                    pname = fg["pattern_name"] if "pattern_name" in fg else "";
                    if pname {
                        setSelectedPatternName(pname);
                    }
                    break;
                }
            }
            if not panelOpen {
                setPanelOpen(True);
            }
            return;
        }
        n_data = rf_node["data"] if rf_node and "data" in rf_node else {};
        bench_op = n_data["bench_op"] if "bench_op" in n_data else "";
        if bench_op and projectName {
            navigate(f"/project/{projectName}/operator-workbench/{bench_op}");
        }
    }

    async def handleAcceptGroup(group_id: str) -> None {
        if not group_id {
            return;
        }
        try {
            await __jacSpawn(
                "UpdateFusionGroup",
                "",
                {
                    "projectName": projectName,
                    "groupId": group_id,
                    "action": "accept",
                    "payload": {}
                }
            );
            await refreshDAG();
        } except Exception as e {
            console.error("handleAcceptGroup error:", e);
        }
    }

    async def handleRejectGroup(group_id: str) -> None {
        if not group_id {
            return;
        }
        try {
            await __jacSpawn(
                "UpdateFusionGroup",
                "",
                {
                    "projectName": projectName,
                    "groupId": group_id,
                    "action": "reject",
                    "payload": {}
                }
            );
            await refreshDAG();
        } except Exception as e {
            console.error("handleRejectGroup error:", e);
        }
    }

    async def handleProposalComplete(new_groups: list) -> None {
        setProposalModalOpen(False);
        setFusionGroups(new_groups);
        setPanelOpen(True);
        await refreshDAG();
    }

    profileBtnLabel = "Profiling..." if profileStatus == "running" else "Re-Profile";
    profileBtnDisabled = profileStatus == "running";

    fg_count = fusionGroups.length if fusionGroups else 0;
    panel_btn_label = "Manage Fusions (" + str(fg_count) + ")"
        if fg_count > 0
        else "Fusions";

    fusions_button = <span/>;
    if fg_count == 0 {
        fusions_button = <GlowButton
            label="Propose Fusions"
            onClick={lambda -> None { setProposalModalOpen(True);}}
        />;
    } else {
        fusions_button = <button
            onClick={lambda -> None { setPanelOpen(not panelOpen);}}
            className="px-3 py-2 bg-[var(--surface-0)] hover:bg-[var(--surface-1)] text-[var(--text-light)] text-xs font-mono uppercase tracking-widest rounded border border-[var(--border)] transition-colors flex items-center gap-2"
            title="Manage Fusions"
        >
            <MenuIcon className="w-4 h-4"/>
            {panel_btn_label}
        </button>;
    }

    panel_el = <span></span>;
    if panelOpen {
        panel_el = <FusionPanel
            fusionGroups={fusionGroups}
            allNodes={allNodes}
            selectedPatternName={selectedPatternName}
            onSelectPattern={setSelectedPatternName}
            onAccept={handleAcceptGroup}
            onReject={handleRejectGroup}
            onPropose={lambda -> None { setProposalModalOpen(True);}}
            onClose={lambda -> None { setPanelOpen(False);}}
        />;
    }

    return
        <div className="flex flex-col h-full bg-[var(--bg-charcoal)]">
            <div
                className="flex-shrink-0 border-b border-[var(--border)] p-4 bg-[var(--bg-panel)] flex justify-between items-center shadow-[0_4px_24px_rgba(0,0,0,0.5)] z-10 relative"
            >
                <div>
                    <h2
                        className="text-xl font-bold text-[var(--text-light)] industrial-font-display tracking-widest uppercase"
                    >
                        Data Flow
                    </h2>
                    <p className="text-[var(--text-muted)] text-sm mt-1 font-mono">
                        Operator execution graph · click node to open workbench
                    </p>
                </div>
                <div className="flex items-center gap-2">
                    {fusions_button}
                    <div className="w-px h-6 bg-[var(--border)] mx-1"></div>
                    <button
                        onClick={startProfile}
                        disabled={profileBtnDisabled}
                        title="Sync Graph / Re-Profile"
                        className={"p-2 rounded border transition-all " + (
                            "bg-[var(--accent-orange)]/20 text-[var(--accent-orange)] border-[var(--accent-orange)]/50 animate-pulse"
                                if profileStatus == "running"
                                else "bg-[var(--surface-0)] hover:bg-[var(--surface-1)] text-[var(--text-muted)] hover:text-[var(--text-light)] border-transparent hover:border-[var(--border)]"
                        )}
                    >
                        <RefreshIcon
                            className={"w-4 h-4 " + (
                                "animate-spin" if profileStatus == "running" else ""
                            )}
                        />
                    </button>
                    <button
                        onClick={exportToMermaid}
                        title="Export to Mermaid"
                        className="p-2 bg-[var(--surface-0)] hover:bg-[var(--surface-1)] text-[var(--text-muted)] hover:text-[var(--text-light)] rounded border border-transparent hover:border-[var(--border)] transition-all"
                    >
                        <DownloadIcon className="w-4 h-4"/>
                    </button>
                </div>
            </div>
            <div className="flex flex-1 overflow-hidden">
                <div className="flex-1 relative react-flow-industrial-theme">
                    <ReactFlow
                        nodes={nodes}
                        edges={edges}
                        onNodesChange={onNodesChange}
                        onEdgesChange={onEdgesChange}
                        onNodeClick={onNodeClick}
                        nodeTypes={nodeTypes_memo}
                        fitView={True}
                        proOptions={{"hideAttribution": True}}
                    >
                        <Controls
                            className="bg-[var(--bg-panel)] border-[var(--border)] fill-[var(--text-light)] shadow-none"
                        />
                        <MiniMap
                            className="bg-[var(--bg-panel)] rounded border border-[var(--border)]"
                            maskColor="rgba(0, 0, 0, 0.4)"
                            zoomable={True}
                            pannable={True}
                        />
                        <Background color="var(--border)" gap={24} size={1}/>
                    </ReactFlow>
                </div>
                {panel_el}
            </div>
            <FusionProposalModal
                isOpen={proposalModalOpen}
                onClose={lambda -> None { setProposalModalOpen(False);}}
                onComplete={handleProposalComplete}
                projectName={projectName}
                config={config}
            />
        </div>;
}
