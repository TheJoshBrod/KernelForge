"""Op Details Component"""

import from react { useState, useEffect, useMemo, useCallback }
import from "reactflow" { default as ReactFlow, Background, Controls, MarkerType, useNodesState, useEdgesState }
import from "recharts" { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine }
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
    has chartData: list = [];
    has minTime: float = 0.0;
    

    
    has isLoading: bool = True;

    def onNodesChange(changes: any) -> None { return; }
    def onEdgesChange(changes: any) -> None { return; }

    # Scale factors for visualization
    X_SCALE = 300;
    Y_SCALE = 150;

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
            
            # Build Tree Structure
            node_map = {};
            children_map = {};
            roots = [];

            for node_data in raw_nodes {
                nid = node_data["id"];
                node_map[nid] = node_data;
                pid = node_data["parent"];
                
                if pid not in children_map {
                    children_map[pid] = [];
                }
                
                if pid != -1 and pid != None {
                    children_map[pid].push(nid);
                } else {
                    roots.push(nid);
                }
                
                # Initialize empty children list for this node if not present
                if nid not in children_map {
                    children_map[nid] = [];
                }
            }
            console.log(f"OpDetails: Found {roots.length} roots", roots);
            console.log("OpDetails: Children map keys", children_map);

            # Layout Algorithm (Recursive)
            visited = {};
            positions = {};
            leaf_y_counter = [0]; # Mutable list to act as non-local

            def traverse_layout(curr_id: int, depth: int) -> float {
                # console.log(f"Traversing {curr_id} at depth {depth}");
                visited[curr_id] = True;
                children = children_map[curr_id];
                
                my_y = 0.0;
                
                if not children or children.length == 0 {
                    # Leaf Node
                    my_y = leaf_y_counter[0] * Y_SCALE;
                    leaf_y_counter[0] += 1;
                } else {
                    # Parent Node - process children first
                    child_ys = [];
                    for child_id in children {
                        child_ys.push(traverse_layout(child_id, depth + 1));
                    }
                    # Center parent vertically relative to children
                    if child_ys.length > 0 {
                        avg_y = 0.0;
                        for y_val in child_ys { avg_y += y_val; }
                        my_y = avg_y / child_ys.length;
                    } else {
                        # Should be covered by 'if not children', but fallback
                         my_y = leaf_y_counter[0] * Y_SCALE;
                         leaf_y_counter[0] += 1;
                    }
                }
                
                positions[curr_id] = {"x": depth * X_SCALE, "y": my_y};
                return my_y;
            }

            # Run layout from all roots
            for root_id in roots {
                traverse_layout(root_id, 0);
            }

            # Generate ReactFlow Nodes and Edges
            new_chart_data = [];
            
            for nid in Object.keys(node_map) {
                node_data = node_map[nid];
                pos = positions[nid];
                if not pos { continue; } # Skip if not reachable (shouldn't happen)
                
                node_id = f"{nid}";
                val = node_data["value"];
                if not val { val = 0.0; }
                
                # Calculate depth from X position
                depth = pos["x"] / X_SCALE;
                new_chart_data.push({"x": depth, "y": val, "id": nid});

                speedup = node_data["speedup_vs_parent"];
                if not speedup { speedup = 0.0; }

                # Label Logic
                label_text = f"ID: {nid}\n{val:.4f}ms";
                if speedup != 1.0 and speedup != 0.0 {
                   label_text += f"\n{speedup:.2f}x"; 
                }

                new_nodes.push({
                    "id": node_id,
                    "position": pos,
                    "data": {
                        "label": label_text
                    },
                    "style": {
                        "background": "#10b981" if speedup > 1.0 else "#ef4444" if speedup < 1.0 else "#3b82f6",
                        "color": "#fff",
                        "width": 140,
                        "fontSize": "12px",
                        "borderRadius": "8px",
                        "border": "1px solid #27272a",
                        "padding": "4px"
                    },
                     "sourcePosition": "right",
                     "targetPosition": "left"
                });

                # Edges
                pid = node_data["parent"];
                if pid != -1 and pid != None {
                    new_edges.push({
                        "id": f"e{pid}-{nid}",
                        "source": f"{pid}",
                        "target": node_id,
                        "type": "smoothstep", 
                        "animated": False,
                        "style": {"stroke": "#71717a", "strokeWidth": 2}
                    });
                }
            }
            
            nodes = new_nodes;
            edges = new_edges;
            chartData = new_chart_data;
            
            # Calculate Min Time
            if new_chart_data.length > 0 {
                 min_val = new_chart_data[0]["y"];
                 for d in new_chart_data {
                     if d["y"] < min_val {
                         min_val = d["y"];
                     }
                 }
                 minTime = min_val;
            }

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
                        <div className="mt-8">
                            <h3 className="text-xl font-semibold border-b border-zinc-800 pb-2 mb-4">
                                Performance vs Depth:
                            </h3>
                            <div className="h-[300px] w-full bg-zinc-900/30 rounded-xl border border-zinc-800 p-4 select-none">
                                <ResponsiveContainer width="100%" height="100%">
                                    <ScatterChart margin={{"top": 20, "right": 20, "bottom": 20, "left": 20}}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                                        <XAxis type="number" dataKey="x" name="Depth" stroke="#888" label={{"value": "Tree Depth", "position": "insideBottom", "offset": -10, "fill": "#888"}} />
                                        <YAxis type="number" dataKey="y" name="Time" unit="ms" stroke="#888" label={{"value": "Time (ms)", "angle": -90, "position": "insideLeft", "fill": "#888"}} />
                                        <Tooltip cursor={{"strokeDasharray": "3 3"}} contentStyle={{"backgroundColor": "#27272a", "borderColor": "#52525b", "color": "#f4f4f5"}} />
                                        <Scatter name="Nodes" data={chartData} fill="#10b981" />
                                        <ReferenceLine y={minTime} stroke="blue" strokeDasharray="3 3" label={{"value": f"Min Found: {minTime:.4f}", "fill": "blue", "position": "insideBottomRight"}} />
                                        <ReferenceLine y={0.0116} stroke="orange" strokeDasharray="3 3" label={{"value": "PyTorch (0.0116)", "fill": "orange", "position": "insideTopRight"}} />
                                    </ScatterChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </div>;
}
