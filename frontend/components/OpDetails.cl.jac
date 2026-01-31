"""Op Details Component"""

import from react { useState, useEffect }
import from "reactflow" { default as ReactFlow, Background, Controls }
import "reactflow/dist/style.css";
import "reactflow/dist/style.css";
import "reactflow/dist/style.css";
import from ..assets.SettingsIcon { SettingsIcon }
import from ..assets.BackIcon { BackIcon }

# Mock data generator logic
def getLayerData(layerName: str) -> dict {
    # Simple deterministic variation based on layer name length
    seed = layerName.length;
    isEven = seed % 2 == 0;
    edges = [];

    baseNodes = [
        {
            "id": "1",
            "position": {"x": 50, "y": 50},
            "data": {"label": f"{layerName}_v1"},
            "style": {"background": "#f97316", "color": "#fff"}
        }
    ];

    if isEven {
        baseNodes = baseNodes.concat(
            [
                {
                    "id": "2",
                    "position": {"x": 250, "y": 100},
                    "data": {"label": "v2 (22ms)"},
                    "style": {"background": "#22c55e", "color": "#fff"}
                },
                {
                    "id": "3",
                    "position": {"x": 450, "y": 150},
                    "data": {"label": "v3 (15ms)"},
                    "style": {"background": "#3b82f6", "color": "#fff"}
                }
            ]
        );
        edges = [
            {"id": "e1-2", "source": "1", "target": "2", "animated": True},
            {"id": "e2-3", "source": "2", "target": "3", "animated": True}
        ];
    } else {
        baseNodes = baseNodes.concat(
            [
                {
                    "id": "2",
                    "position": {"x": 250, "y": 150},
                    "data": {"label": "v2 (19ms)"},
                    "style": {"background": "#22c55e", "color": "#fff"}
                },
                {
                    "id": "3",
                    "position": {"x": 450, "y": 50},
                    "data": {"label": "v3 (17ms)"},
                    "style": {"background": "#22c55e", "color": "#fff"}
                },
                {
                    "id": "4",
                    "position": {"x": 650, "y": 120},
                    "data": {"label": "v4 (12ms)"},
                    "style": {"background": "#3b82f6", "color": "#fff"}
                }
            ]
        );
        edges = [
            {"id": "e1-2", "source": "1", "target": "2", "animated": True},
            {"id": "e2-3", "source": "2", "target": "3", "animated": True},
            {"id": "e3-4", "source": "3", "target": "4", "animated": True}
        ];
    }

    return {"nodes": baseNodes, "edges": edges};
}

def:pub OpDetails(projectName: str, opName: str, onNavigate: any) -> any {
    # Generate some layers based on opName (simulated)
    has layers: list = [
        f"{opName}1",
        f"{opName}2",
        f"{opName}3",
        f"{opName}4",
        f"{opName}5"
    ];
    has selectedLayer: str = layers[0];

    # State for nodes and edges
    has nodes: list = [];
    has edges: list = [];

    # Effect to update graph when selectedLayer changes
    useEffect(
        lambda -> None { if selectedLayer {
            data = getLayerData(selectedLayer);
            nodes = data["nodes"];
            edges = data["edges"];
        }},
        [selectedLayer]
    );

    # Handle Escape key
    useEffect(
        lambda -> any { def handleKeyDown(event: any) -> None {
            if event.key == "Escape" {
                onNavigate(f"/project/{projectName}");
            }
        } window.addEventListener("keydown", handleKeyDown);return lambda -> None { window.removeEventListener(
            "keydown", handleKeyDown
        );}; },
        [projectName]
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
                        <div
                            className="h-[500px] bg-zinc-900/30 rounded-xl border border-zinc-800 p-4 relative"
                        >
                            <h4
                                className="absolute top-4 left-4 text-sm font-medium text-zinc-400 z-10"
                            >
                                Y-Axis: Execution Time (Lower is better) | X-Axis: Generation
                            </h4>
                            <ReactFlow
                                nodes={nodes}
                                edges={edges}
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
                    </div>
                </div>
            </main>
        </div>;
}
