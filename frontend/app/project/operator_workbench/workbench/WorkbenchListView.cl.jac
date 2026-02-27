"""Workbench List View Component"""

def:pub WorkbenchListView(props: any) -> Any {
    viewMode = props.viewMode;
    sortedNodes = props.sortedNodes;
    selectedNodeId = props.selectedNodeId;
    onNodeSelect = props.onNodeSelect;
    torchDisplayMs = props.torchDisplayMs;

    if viewMode != "list" {
        return <span/>;
    }

    pytorch_el = <span/>;
    if torchDisplayMs and torchDisplayMs > 0 {
        pytorch_el = (
            <div
                key="pytorch-baseline"
                className="p-2.5 rounded-md border border-indigo-500/30 bg-indigo-500/10 shadow-[0_0_15px_rgba(99,102,241,0.05)] flex items-center gap-4 group"
            >
                <div className="text-[var(--text-muted)] font-mono text-[11px] w-8 text-right">
                    --
                </div>
                <div className="font-mono text-sm font-semibold text-indigo-300">
                    PyTorch Baseline
                </div>
                <div className="font-mono text-indigo-300 text-sm font-semibold flex-1">
                    <span>{Number(torchDisplayMs).toFixed(6)}</span>
                    <span className="text-[11px] ml-1 opacity-70">ms</span>
                </div>
            </div>
        );
    }

    listViewItems = [];
    pytorch_inserted = False;

    idx = 0;
    for n in sortedNodes {
        val = Number(n["value"] or 0);
        if not pytorch_inserted and torchDisplayMs and torchDisplayMs > 0 and val > torchDisplayMs {
            listViewItems.push(pytorch_el);
            pytorch_inserted = True;
        }

        idx += 1;
        listViewItems.push(
            <div
                key={n["id"]}
                className={"p-2.5 rounded-md border cursor-pointer transition-all flex items-center gap-4 group " + (
                    "border-[var(--accent)]/50 bg-[var(--accent)]/10 shadow-[0_0_15px_rgba(253,203,110,0.1)]"
                    if f"{n['id']}" == selectedNodeId
                    else "border-[var(--border)] bg-[var(--surface-0)]/50 hover:border-zinc-600 hover:bg-[var(--surface-1)]/50"
                )}
                onClick={lambda event: any  -> None { onNodeSelect(f"{n['id']}");}}
            >
                <div className="text-[var(--text-muted)] font-mono text-[11px] w-8 text-right">
                    {"#" + str(idx)}
                </div>
                <div
                    className={"font-mono text-sm font-semibold " + (
                        "text-[var(--accent)]"
                        if f"{n['id']}" == selectedNodeId
                        else "text-[var(--text-light)]"
                    )}
                >
                    Kernel{" " + n["id"]}:
                </div>
                <div
                    className={"font-mono text-sm font-semibold flex-1 " + (
                        "text-[var(--accent)]"
                        if f"{n['id']}" == selectedNodeId
                        else "text-zinc-300"
                    )}
                >
                    <span>
                        {Number(n["value"] or 0).toFixed(6)}
                    </span>
                    <span className="text-[11px] ml-1 opacity-70">
                        ms
                    </span>
                </div>
            </div>
        );
    }

    if not pytorch_inserted and torchDisplayMs and torchDisplayMs > 0 {
        listViewItems.push(pytorch_el);
    }

    return
        <div className="h-full overflow-y-auto p-4 pt-14 space-y-2 custom-scrollbar">
            {listViewItems}{<div
                className="text-center py-10 text-[var(--text-muted)] italic text-sm"
            >
                No kernels found.
            </div>
            if sortedNodes.length == 0
            else <span/>}
        </div>;
}
