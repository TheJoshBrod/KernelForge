"""Op Details Component"""

import from react { useState, useEffect, useMemo, useCallback }
import from "reactflow" { default as ReactFlow, Background, Controls, MarkerType, useNodesState, useEdgesState }
import "reactflow/dist/style.css";
import from ..assets.SettingsIcon { SettingsIcon }
import from ..assets.BackIcon { BackIcon }

def:pub OpDetails(projectName: str, opName: str, onNavigate: any, getOpDataWalker: any) -> any {
    # Layout State
    has layers: list = [
        opName,
        f"{opName}_v1",
        f"{opName}_v2"
    ];
    has selectedLayer: str = opName;
    
    # State for graph data - Idiomatic Jac
    has nodes: list = [];
    has edges: list = [];
    has isLoading: bool = True;

    def onNodesChange(changes: any) -> None { return; }
    def onEdgesChange(changes: any) -> None { return; }

    # Scale factors for visualization
    X_SCALE = 150;
    Y_SCALE = 50;

    # Fetch data on layer change
    async def fetchData() -> None {
        try {
            console.log(f"OpDetails: fetching data for {selectedLayer}...");
            isLoading = True;
            
            # Using selectedLayer as the opName for the walker to allow switching
            # Note: For now, this might only work if the folder name matches selectedLayer
            # Since we have a hardcoded path in the walker, it might ignore this arg,
            # but we pass it anyway for future dynamic support.
            data = await getOpDataWalker(projectName, selectedLayer);
            console.log("OpDetails: received data", data);
            
            raw_nodes = data["nodes"];
            if not raw_nodes {
                raw_nodes = [];
            }
            
            if raw_nodes.length == 0 {
                console.log("OpDetails: no nodes found");
                nodes = [];
                edges = [];
                isLoading = False;
                return;
            }

            new_nodes = [];
            new_edges = [];
            
            # Process nodes
            processed_count = 0;
            for node in raw_nodes {
                processed_count += 1;
                node_id = f"{node['id']}";
                # Y axis: Time
                val = node["value"];
                if not val { val = 0.0; }
                y_pos = (val) * 500; 
                
                # X axis
                x_pos = node["id"] * 200;

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
            
            nodes = new_nodes;
            edges = new_edges;

        } except Exception as e {
            console.error("Failed to fetch graph data:", e);
        } finally {
            isLoading = False;
        }
    }

    # Fetch data on selection change
    useEffect(
        lambda -> None {
            fetchData();
        },
        [selectedLayer]
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


            <div
                className="w-1/4 border-r border-zinc-900 flex flex-col p-4 space-y-6"
            >
                <div>
                    <h2 className="text-xl font-bold mb-4 text-zinc-100">
                        Layers:
                    </h2>
                    <div className="flex gap-2">
                        <input
                            type="text"
                            placeholder="Search..."
                            className="flex-1 bg-zinc-900 border border-zinc-800 rounded px-3 py-2 text-sm focus:outline-none focus:border-zinc-700 transition-colors"
                        />
                        <button
                            className="bg-zinc-800 hover:bg-zinc-700 hover:text-white px-4 py-2 rounded text-sm font-medium transition-colors"
                        >
                            Filter
                        </button>
                    </div>
                </div>
                <div
                    className="flex-1 overflow-y-auto space-y-2 pr-2 custom-scrollbar"
                >
                    {[
                        <button
                            key={layer}
                            onClick={(
                                lambda l: str  -> any { return lambda -> None { selectedLayer = l;}; }
                            )(
                                layer
                            )}
                            className={"w-full text-left border rounded px-4 py-3 text-sm font-mono transition-all " + (
                                "bg-zinc-800 border-zinc-600 text-white"
                                if selectedLayer == layer
                                else "bg-zinc-900/50 hover:bg-zinc-800 border-zinc-800/50 text-zinc-300"
                            )}
                        >
                            {layer}
                        </button> for layer in layers
                    ]}
                </div>
                <div className="space-y-4 pt-4 border-t border-zinc-900">
                    <button
                        className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 rounded shadow-lg shadow-blue-900/20 transition-all"
                    >
                        New Layer
                    </button>
                </div>
            </div>


            <main className="w-3/4 p-8 overflow-y-auto">
                <h1 className="text-3xl font-bold text-center mb-12 tracking-tight">
                    Optimizing Layer:
                    <span className="text-emerald-400">
                        {selectedLayer}
                    </span>
                </h1>
                <div className="grid grid-cols-1 gap-8 mb-12">
                    <div className="space-y-6">
                        <h3
                            className="text-xl font-semibold border-b border-zinc-800 pb-2"
                        >
                            Kernel Refinement Graph:
                        </h3>
                        
                        {<div
                            className="h-[500px] bg-zinc-900/30 rounded-xl border border-zinc-800 p-4 relative"
                        >
                            <h4
                                className="absolute top-4 left-4 text-sm font-medium text-zinc-400 z-10 bg-zinc-950/50 px-2 py-1 rounded"
                            >
                                Y-Axis: Execution Time (Higher is Better) | X-Axis: Generation
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
                                panOnDrag={True}
                                zoomOnScroll={True}
                                zoomOnPinch={True}
                                zoomOnDoubleClick={True}
                                proOptions={{"hideAttribution": True}}
                            >
                                <Background color="#27272a" gap={16} />
                                <Controls
                                    className="bg-zinc-800 border-zinc-700 fill-zinc-400"
                                />
                            </ReactFlow>
                        </div>
                        if not isLoading else
                        <div className="h-[500px] flex items-center justify-center text-zinc-500 animate-pulse bg-zinc-900/10 rounded-xl border border-zinc-800">
                            Loading optimization data for {selectedLayer}...
                        </div>}
                    </div>
                </div>
            </main>
        </div>;
}
