"""Workbench Node Details Component"""

import from .....components.ui.Icons { ShieldCheckIcon }

def:pub WorkbenchNodeDetails(props: any) -> Any {
    selectedNodeId = props.selectedNodeId;
    selectedNodeData = props.selectedNodeData;
    torchDisplayMs = props.torchDisplayMs;

    torch_display_text = "--";
    if torchDisplayMs and torchDisplayMs > 0 {
        torch_display_text = f"{Number(torchDisplayMs).toFixed(9)} ms";
    }

    return
        <div className="flex-1 p-6 overflow-y-auto">
            <div
                className="flex justify-between items-center mb-4 border-b border-zinc-800 pb-2"
            >
                <h3
                    className="text-lg font-bold text-white flex items-center gap-2"
                >
                    <ShieldCheckIcon className="w-5 h-5 text-teal-400"/>
                    Kernel {" " + selectedNodeData["id"]} Details
                </h3>
                <span className="font-mono text-zinc-400 text-xs">
                    {Reflect.construct(
                        Date, [selectedNodeData["timestamp"] * 1000]
                    ).toLocaleString()
                    if selectedNodeData and "timestamp" in selectedNodeData and selectedNodeData["timestamp"] > 0.0
                    else ""}
                </span>
            </div>
            {<div className="space-y-4">
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
                Select a kernel from the tree to view details.
            </div>}
        </div>;
}
