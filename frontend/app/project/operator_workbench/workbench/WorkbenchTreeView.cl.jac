"""Workbench Tree View Component"""

import from reactflow { ReactFlow, Background }

def:pub WorkbenchTreeView -> Any {
    has displayNodes: list,
        displayEdges: list,
        onNodesChange: Any,
        onEdgesChange: Any,
        onNodeClick: Any,
        hasTree: bool,
        isOptimizing: bool;

    if not (hasTree or isOptimizing) {
        return
            <div
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
            </div>;
    }

    return
        <ReactFlow
            nodes={displayNodes}
            edges={displayEdges}
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
        </ReactFlow>;
}
