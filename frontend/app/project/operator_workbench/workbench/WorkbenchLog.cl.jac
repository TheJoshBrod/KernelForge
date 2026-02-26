"""Workbench Log Panel Component"""

import from react { useState }
import from .....components.ui.Icons { ActivityIcon }

def:pub WorkbenchLog -> Any {
    has selectedNodeId: Any,
        rawNodes: dict,
        nodes: list,
        edges: list;

    log_collapsed_state = useState(False);
    logCollapsed = log_collapsed_state[0];
    setLogCollapsed = log_collapsed_state[1];

    log_items = [];
    if selectedNodeId and nodes.length > 0 and edges.length > 0 {
        parent_map = {};
        for e in edges {
            parent_map[e.target] = e.source;
        }
        curr = selectedNodeId;
        while curr {
            n_data = rawNodes[curr];
            if n_data {
                log_items.unshift(n_data);
            }
            curr = parent_map[curr] if curr in parent_map else None;
        }
    }

    logViewItems = [];
    for item in log_items {
        logViewItems.push(
            <div
                key={item["id"]}
                className={"p-3 rounded border text-xs " + (
                    "border-orange-500/50 bg-orange-500/10"
                    if f"{item['id']}" == selectedNodeId
                    else "border-zinc-800 bg-zinc-900/50"
                )}
                ref={lambda el: any  -> None { if f"{item['id']}" == selectedNodeId
                and el {
                    el.scrollIntoView({"behavior": "smooth", "block": "nearest"});
                }}}
            >
                <div className="flex justify-between mb-1">
                    <span className="font-mono font-bold text-orange-300">
                        Kernel{" " + item["id"]}
                    </span>
                    <span className="text-teal-300 font-mono">
                        {Number(item["value"] or 0).toFixed(4)}
                    </span>
                </div>
                <div className="text-zinc-400">
                    {item["improvement_description"]}
                </div>
            </div>
        );
    }

    return
        <div
            className={"flex flex-col bg-zinc-900/20 " + (
                "h-10" if logCollapsed else "h-[40%]"
            )}
        >
            <div
                className={"h-10 bg-zinc-900/50 flex items-center px-4 justify-between " + (
                    "border-b border-zinc-800" if not logCollapsed else ""
                )}
            >
                <span
                    className="text-xs font-bold text-zinc-500 uppercase flex items-center gap-2"
                >
                    <ActivityIcon className="w-4 h-4"/>
                    Iteration Log
                </span>
                <button
                    className="text-[11px] text-zinc-400 hover:text-zinc-200 px-2 py-1 rounded border border-zinc-700 bg-zinc-900/50"
                    onClick={lambda -> None { setLogCollapsed(not logCollapsed);}}
                >
                    {"Expand" if logCollapsed else "Collapse"}
                </button>
            </div>
            {<div
                className="flex-1 overflow-y-auto p-4 space-y-2 flex flex-col-reverse"
            >
                {logViewItems}
            </div>
            if not logCollapsed
            else <span></span>}
        </div>;
}
