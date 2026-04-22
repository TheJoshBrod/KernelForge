"""Workbench Node Details Component"""

import from .....components.ui.Icons { ShieldCheckIcon }

def:pub WorkbenchNodeDetails(props: any) -> Any {
    selectedNodeId = props.selectedNodeId;
    selectedNodeData = props.selectedNodeData;
    timing_metric = (
        str(selectedNodeData["timing_metric"])
        if selectedNodeData and "timing_metric" in selectedNodeData and selectedNodeData["timing_metric"]
        else "mean_time_ms"
    );
    value_label = (
        "Average Runtime (ms)"
        if timing_metric == "mean_time_ms"
        else "Stored Runtime (legacy)"
    );
    subtree_label = (
        "Best Subtree Runtime (ms)"
        if timing_metric == "mean_time_ms"
        else "Best Subtree Value"
    );

    return
        <div className="flex-1 flex flex-col min-h-0">
            <div
                className="h-10 bg-zinc-900/50 border-b border-zinc-800 flex items-center px-4 justify-between flex-shrink-0"
            >
                <span
                    className="text-xs font-semibold text-zinc-500 uppercase tracking-wider flex items-center gap-2"
                >
                    <ShieldCheckIcon className="w-4 h-4 text-[var(--accent)]"/>
                    Node Details
                </span>
                <span className="text-xs font-mono text-orange-300">
                    Kernel {" " + (selectedNodeId or "?")}
                </span>
            </div>
            <div className="flex-1 p-5 overflow-y-auto custom-scrollbar">
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
                        {value_label}
                    </span>
                    <span className="font-mono text-sm font-medium text-[var(--text-light)]">
                        {Number(selectedNodeData["value"] or 0).toFixed(6)}
                    </span>
                </div>
                <div className="flex justify-between items-center bg-[var(--surface-0)]/50 p-2.5 rounded border border-[var(--border)]">
                    <span className="text-[var(--text-muted)] text-xs uppercase font-medium">
                        {subtree_label}
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
            </div>
        </div>;
}
