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
                className="p-3 rounded-lg border border-blue-500/50 bg-blue-500/10 shadow-[0_0_15px_rgba(59,130,246,0.1)] flex items-center gap-4 group"
            >
                <div className="text-zinc-500 font-mono text-xs w-8 text-right">
                    --
                </div>
                <div className="font-mono text-sm font-bold text-blue-300">
                    PyTorch Baseline
                </div>
                <div className="font-mono text-blue-300 text-sm font-bold flex-1">
                    <span>{Number(torchDisplayMs).toFixed(6)}</span>
                    <span>{" ms"}</span>
                </div>
                <div className="text-zinc-600 text-xs font-mono w-40 text-right">
                    --
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
                className={"p-3 rounded-lg border cursor-pointer transition-all flex items-center gap-4 group " + (
                    "border-orange-500/80 bg-orange-500/10 shadow-[0_0_15px_rgba(225,112,85,0.18)]"
                    if f"{n['id']}" == selectedNodeId
                    else "border-zinc-800 bg-zinc-900/30 hover:border-zinc-700 hover:bg-zinc-900/80"
                )}
                onClick={lambda event: any  -> None { onNodeSelect(f"{n['id']}");}}
            >
                <div className="text-zinc-500 font-mono text-xs w-8 text-right">
                    {"#" + str(idx)}
                </div>
                <div
                    className={"font-mono text-sm font-bold " + (
                        "text-orange-300"
                        if f"{n['id']}" == selectedNodeId
                        else "text-zinc-300"
                    )}
                >
                    Kernel{" " + n["id"]}:
                </div>
                <div
                    className="font-mono text-teal-300 text-sm font-bold flex-1"
                >
                    <span>
                        {Number(n["value"] or 0).toFixed(6)}
                    </span>
                    <span>
                        {" ms"}
                    </span>
                </div>
                <div className="text-zinc-600 text-xs font-mono w-40 text-right">
                    {Reflect.construct(Date, [n["timestamp"] * 1000]).toLocaleString()
                    if "timestamp" in n and n["timestamp"] > 0.0
                    else ""}
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
                className="text-center py-10 text-zinc-600 italic text-sm"
            >
                No kernels found.
            </div>
            if sortedNodes.length == 0
            else <span/>}
        </div>;
}
