"""Workbench Node Details Component"""

import from .....components.ui.Icons { ShieldCheckIcon }

def:pub WorkbenchNodeDetails(props: any) -> Any {
    selectedNodeId = props.selectedNodeId;
    selectedNodeData = props.selectedNodeData;

    return
        <div className="flex-1 p-5 overflow-y-auto custom-scrollbar">
            <div
                className="flex justify-between items-center mb-4 border-b border-[var(--border)] pb-3"
            >
                <h3
                    className="text-sm font-semibold text-[var(--text-light)] uppercase tracking-wider flex items-center gap-2"
                >
                    <ShieldCheckIcon className="w-5 h-5 text-[var(--accent)]"/>
                    Kernel {" " + selectedNodeData["id"]} Details
                </h3>
            </div>
            {<div className="space-y-4">
                <div className="flex justify-between items-center bg-[var(--surface-0)]/50 p-2.5 rounded border border-[var(--border)]">
                    <span className="text-[var(--text-muted)] text-xs uppercase font-medium">
                        Visits
                    </span>
                    <span className="font-mono text-sm font-medium text-[var(--text-light)]">
                        {selectedNodeData["visits"]}
                    </span>
                </div>
                <div className="flex justify-between items-center bg-[var(--surface-0)]/50 p-2.5 rounded border border-[var(--border)]">
                    <span className="text-[var(--text-muted)] text-xs uppercase font-medium">
                        Reward
                    </span>
                    <span className="font-mono text-sm font-medium text-[var(--text-light)]">
                        {Number(selectedNodeData["value"] or 0).toFixed(6)}
                    </span>
                </div>
                <div className="flex justify-between items-center bg-[var(--surface-0)]/50 p-2.5 rounded border border-[var(--border)]">
                    <span className="text-[var(--text-muted)] text-xs uppercase font-medium">
                        Best Subtree
                    </span>
                    <span className="font-mono text-sm font-medium text-[var(--text-light)]">
                        {Number(selectedNodeData["best_subtree_value"] or 0).toFixed(
                            6
                        )}
                    </span>
                </div>
                <div className="flex justify-between items-center bg-[var(--surface-0)]/50 p-2.5 rounded border border-[var(--border)]">
                    <span className="text-[var(--text-muted)] text-xs uppercase font-medium">
                        Generated
                    </span>
                    <span className="font-mono text-sm font-medium text-[var(--text-light)]">
                        {Reflect.construct(
                            Date, [selectedNodeData["timestamp"] * 1000]
                        ).toLocaleString()
                        if selectedNodeData and "timestamp" in selectedNodeData and selectedNodeData["timestamp"] > 0.0
                        else "—"}
                    </span>
                </div>
            </div>
            if selectedNodeId
            else <div className="text-[var(--text-muted)] text-xs italic text-center py-4">
                Select a kernel from the tree to view details.
            </div>}
        </div>;
}
