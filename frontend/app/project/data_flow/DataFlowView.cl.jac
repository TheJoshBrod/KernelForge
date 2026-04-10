"""Data Flow View Component"""

import from react { useState, useEffect, useRef }
import from reactflow {
    ReactFlow,
    Background,
    Controls,
    MiniMap,
    useNodesState,
    useEdgesState,
    addEdge
}
import from "react-router-dom" { useParams }
import from "./dagreWrap.js" { createGraph, layoutGraph as dagreLayout }
import from ....components.ui.GlowButton { GlowButton }

def:pub DataFlowView -> Any {
    params = useParams();
    projectName = params.projectName;

    nodes_state = useNodesState([]);
    edges_state = useEdgesState([]);
    profile_state = useState("idle");
    profileStatus = profile_state[0];
    setProfileStatus = profile_state[1];
    pollRef = useRef(None);

    nodes = nodes_state[0];
    setNodes = nodes_state[1];
    onNodesChange = nodes_state[2];

    edges = edges_state[0];
    setEdges = edges_state[1];
    onEdgesChange = edges_state[2];

    def layoutGraph(dag_data: dict) -> None {
        g = createGraph();
        g.setGraph({"rankdir": "TB"});
        g.setDefaultEdgeLabel(lambda -> dict { return {}; });

        if not dag_data or not ("nodes" in dag_data) {
            return;
        }

        nodes_list = dag_data["nodes"] if "nodes" in dag_data else [];
        edges_list = dag_data["edges"] if "edges" in dag_data else [];

        for n in nodes_list {
            g.setNode(str(n["id"]), {"width": 180, "height": 60});
        }
        for e in edges_list {
            g.setEdge(str(e["source"]), str(e["target"]));
        }

        dagreLayout(g);

        new_nodes = [];
        for n in nodes_list {
            node_id = str(n["id"]);
            pos = g.node(node_id);
            node_label = f"{n['op']}\n({node_id})";
            new_nodes.append(
                {
                    "id": node_id,
                    "position": {"x": pos.x - 90, "y": pos.y - 30},
                    "data": {"label": node_label}
                }
            );
        }

        new_edges = [];
        idx = 0;
        for e in edges_list {
            new_edges.append(
                {
                    "id": f"e_{idx}",
                    "source": str(e["source"]),
                    "target": str(e["target"]),
                    "type": "smoothstep"
                }
            );
            idx = idx + 1;
        }

        setNodes(new_nodes);
        setEdges(new_edges);
    }

    async def refreshDAG -> None {
        try {
            if projectName {
                rel_path = f"kernels/projects/{projectName}/io/dag.json";
                resp = await __jacSpawn(
                    "ReadRepoFile",
                    "",
                    {"projectName": projectName, "relPath": rel_path}
                );
                data = resp.reports[0]
                    if resp and resp.reports and resp.reports.length > 0
                    else {};
                if data and "success" in data and data["success"] {
                    content = data["content"];
                    dag_data = JSON.parse(content);
                    layoutGraph(dag_data);
                    if pollRef.current {
                        clearInterval(pollRef.current);
                        pollRef.current = None;
                        setProfileStatus("idle");
                    }
                } else {
                    console.log("No DAG found or profile needs to be re-run");
                }
            }
        } except Exception as e {
            console.error("Failed to fetch DAG:", e);
        }
    }

    useEffect(lambda -> None { refreshDAG();}, [projectName]);

    def onConnect(conn_params: dict) -> None {
        setEdges(lambda eds: any  -> any { return addEdge(conn_params, eds); });
    }

    def exportToMermaid -> None {
        mermaid = "graph TD;\n";
        for edge in edges {
            source_id = edge["source"];
            target_id = edge["target"];

            s_label = source_id;
            t_label = target_id;

            for n in nodes {
                if n["id"] == source_id {
                    s_label = n["data"]["label"].replace("\n", " ");
                }
                if n["id"] == target_id {
                    t_label = n["data"]["label"].replace("\n", " ");
                }
            }

            mermaid = mermaid + f"    {source_id}[\"{s_label}\"] --> {target_id}[\"{t_label}\"];\n";
        }

        navigator.clipboard.writeText(mermaid);
        alert("Mermaid code copied to clipboard!\n\n" + mermaid);
    }

    async def startProfile -> None {
        try {
            setProfileStatus("running");
            await __jacSpawn(
                "StartProfile", "", {"projectName": projectName, "force": True}
            );
            pollRef.current = setInterval(
                lambda -> None {
                    refreshDAG();
                },
                10000
            );
            setTimeout(
                lambda -> None {
                    if pollRef.current {
                        clearInterval(pollRef.current);
                        pollRef.current = None;
                        setProfileStatus("idle");
                    }
                },
                120000
            );
        } except Exception as e {
            console.error("Failed to start profile:", e);
            setProfileStatus("idle");
        }
    }

    profileBtnLabel = "Profiling..." if profileStatus == "running" else "Re-Profile";
    profileBtnDisabled = profileStatus == "running";

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
                        Visualizing compute graph natively
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    <button
                        onClick={startProfile}
                        disabled={profileBtnDisabled}
                        className="px-3 py-2 bg-[var(--surface-0)] hover:bg-[var(--surface-1)] text-[var(--text-light)] text-xs font-mono uppercase tracking-widest rounded border border-[var(--border)] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {profileBtnLabel}
                    </button>
                    <button
                        onClick={refreshDAG}
                        className="px-3 py-2 bg-[var(--surface-0)] hover:bg-[var(--surface-1)] text-[var(--text-light)] text-xs font-mono uppercase tracking-widest rounded border border-[var(--border)] transition-colors"
                    >
                        Refresh
                    </button>
                    <GlowButton color="accent" onClick={exportToMermaid}>
                        Export to Mermaid
                    </GlowButton>
                </div>
            </div>
            <div className="flex-1 relative react-flow-industrial-theme">
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    fitView={True}
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
        </div>;
}
