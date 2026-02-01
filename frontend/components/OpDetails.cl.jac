"""Op Details Component"""

import from react { useState, useEffect, useMemo, useCallback }
import from "reactflow" { default as ReactFlow, Background, Controls, MarkerType, useNodesState, useEdgesState }
import "reactflow/dist/style.css";
import from ..assets.SettingsIcon { SettingsIcon }
import from ..assets.BackIcon { BackIcon }

def:pub OpDetails(projectName: str, opName: str, onNavigate: any, getOpDataWalker: any) -> any {
    # State for graph data - Idiomatic Jac
    has nodes: list = [];
    has edges: list = [];
    has isLoading: bool = True;
    
    def onNodesChange(changes: any) -> None { return; }
    def onEdgesChange(changes: any) -> None { return; }

    console.log("OpDetails Render: isLoading =", isLoading, "nodes len:", nodes.length);

    # Scale factors for visualization
    X_SCALE = 150;
    Y_SCALE = 50;

    # Track mounting
    useEffect(lambda -> None {
        console.log("OpDetails: MOUNTED");
    }, []);

    # Fetch data on mount
    async def fetchData() -> None {
        try {
            console.log("OpDetails: fetching data...");
            isLoading = True;
            data = await getOpDataWalker(projectName, opName);
            console.log("OpDetails: received data", data);
            
            raw_nodes = data["nodes"];
            if not raw_nodes {
                raw_nodes = [];
            }
            
            if raw_nodes.length == 0 {
                console.log("OpDetails: no nodes found");
                nodes = [];
                edges = [];
                console.log("Setting isLoading False (empty nodes)");
                isLoading = False;
                return;
            }

            new_nodes = [];
            new_edges = [];
            
            # Process nodes
            processed_count = 0;
            for node in raw_nodes {
                processed_count += 1;
                console.log(f"Processing node {processed_count}");
                node_id = f"{node['id']}";
                # Y axis: Time
                val = node["value"];
                if not val { val = 0.0; }
                y_pos = (val) * 500; 
                
                # X axis
                x_pos = node["id"] * 200;

                improvement_desc = node["improvement_description"];
                if not improvement_desc { improvement_desc = ""; }
                
                speedup = node["speedup_vs_parent"];
                if not speedup { speedup = 0.0; }

                new_nodes.push({
                    "id": node_id,
                    "position": {"x": x_pos, "y": y_pos},
                    "data": {
                        "label": f"ID: {node_id}\n{val}ms"
                    },
                    "style": {
                        "background": "#10b981" if speedup > 1.0 else "#ef4444",
                        "color": "#fff",
                        "width": 150,
                        "fontSize": "12px"
                    }
                });

                # Process edges
                parent_id = node["parent"];
                if parent_id != -1 and parent_id != None {
                        new_edges.push({
                        "id": f"e{parent_id}-{node_id}",
                        "source": f"{parent_id}",
                        "target": node_id,
                        "animated": True,
                        "style": {"stroke": "#52525b"}
                    });
                }
            }
            
            console.log("Setting nodes and edges:", new_nodes.length, new_edges.length);
            if new_nodes.length > 0 {
                console.log("First node:", new_nodes[0]);
            }
            nodes = new_nodes;
            edges = new_edges;

        } except Exception as e {
            console.error("Failed to fetch graph data:", e);
        } finally {
            console.log("Finally: Setting isLoading False");
            isLoading = False;
        }
    }

    # Fetch data on mount
    useEffect(
        lambda -> None {
            fetchData();
        },
        [projectName, opName]
    );

    return
        <div className="flex min-h-screen bg-zinc-950 text-zinc-100 font-sans">
            <div className="absolute top-2 right-6">
                <a
                    onClick={lambda -> None { onNavigate("/settings");}}
                    className="flex items-center justify-center p-2 rounded-full hover:bg-zinc-800 transition-colors text-zinc-400 hover:text-white cursor-pointer"
                    aria-label="Settings"
                >
                    <SettingsIcon />
                </a>
            </div>
            <div className="absolute top-2 right-16 z-10">
                <a
                    onClick={lambda -> None { onNavigate(f"/project/{projectName}");}}
                    className="flex items-center justify-center p-2 rounded-full hover:bg-zinc-800 transition-colors text-zinc-400 hover:text-white bg-zinc-950/50 backdrop-blur-sm cursor-pointer"
                    aria-label="Go Back"
                >
                    <BackIcon />
                </a>
            </div>

            <main className="w-full h-screen p-8 flex flex-col">
                <h1 className="text-3xl font-bold text-center mb-4 tracking-tight">
                    Optimization Graph: <span className="text-emerald-400">{opName}</span>
                </h1>
                
                {<div className="flex-1 rounded-xl border border-zinc-800 bg-zinc-900/30 relative overflow-hidden">
                    <h4 className="absolute top-4 left-4 text-sm font-medium text-zinc-400 z-10 bg-zinc-950/50 px-2 py-1 rounded">
                        Y-Axis: Execution Time (ms) | X-Axis: Iteration ID
                    </h4>
                     <ReactFlow
                        nodes={nodes}
                        edges={edges}
                        onNodesChange={onNodesChange}
                        onEdgesChange={onEdgesChange}
                        fitView={True}
                        className="bg-zinc-900/30"
                        nodesConnectable={False}
                        nodesDraggable={True} 
                    >
                        <Background color="#27272a" gap={16} />
                        <Controls className="bg-zinc-800 border-zinc-700 fill-zinc-400" />
                    </ReactFlow>
                </div>
                if not isLoading else 
                <div className="flex-1 flex items-center justify-center text-zinc-500 animate-pulse flex-col">
                    Loading optimization data...
                    <br />
                    <span className="text-xs text-zinc-600 mt-2">
                        Debug: isLoading is {f"{isLoading}"}
                    </span>
                    <span className="text-xs text-zinc-700 mt-1 max-w-md text-center">
                        Accessing: kernels/optimized/NVIDIA_GeForce_GTX_1660_Ti_egr6dqhOEU/...
                    </span>
                </div>}
            </main>
        </div>;
}
