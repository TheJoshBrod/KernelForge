"""Workbench Log Panel Component"""

import from react { useState }
import from .....components.ui.Icons { ActivityIcon }

def:pub WorkbenchLog(props: any) -> Any {
    selectedNodeId = props.selectedNodeId;
    rawNodes = props.rawNodes;
    nodes = props.nodes;
    edges = props.edges;

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
    idx = 0;
    for item in log_items {
        is_even = (idx % 2) == 0;
        base_bg = "border-[var(--border)] bg-[var(--surface-0)]/50" if is_even else "border-[var(--border)] bg-[var(--surface-1)]/50";

        logViewItems.push(
            <div
                key={item["id"]}
                className={"p-2.5 rounded-md border transition-all " + (
                    "border-[var(--accent)]/50 bg-[var(--accent)]/10 shadow-[0_0_15px_rgba(253,203,110,0.1)]"
                    if f"{item['id']}" == selectedNodeId
                    else base_bg
                )}
                ref={lambda el: any  -> None { if f"{item['id']}" == selectedNodeId
                and el {
                    el.scrollIntoView({"behavior": "smooth", "block": "nearest"});
                }}}
            >
                <div className="flex justify-between items-center mb-1">
                    <span className={"font-mono text-sm font-semibold " + ("text-[var(--accent)]" if f"{item['id']}" == selectedNodeId else "text-[var(--text-light)]")}>
                        Kernel{" " + item["id"]}
                    </span>
                    <div className={"font-mono text-sm font-semibold " + ("text-[var(--accent)]" if f"{item['id']}" == selectedNodeId else "text-[var(--text-light)]")}>
                        <span>{Number(item["value"] or 0).toFixed(4)}</span>
                    </div>
                </div>
                <div className="text-[var(--text-muted)] text-[11px] leading-relaxed">
                    {item["improvement_description"]}
                </div>
            </div>
        );
        idx += 1;
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
                    className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider flex items-center gap-2"
                >
                    <ActivityIcon className="w-4 h-4 text-[var(--accent)]"/>
                    Iteration Log
                </span>
                <button
                    className="text-[11px] text-[var(--text-muted)] hover:text-[var(--text-light)] px-2 py-1 rounded border border-[var(--border)] bg-[var(--surface-0)]/50 transition-colors"
                    onClick={lambda -> None { setLogCollapsed(not logCollapsed);}}
                >
                    {"Expand" if logCollapsed else "Collapse"}
                </button>
            </div>
            {<div
                className="flex-1 overflow-y-auto p-4 space-y-2 flex flex-col-reverse custom-scrollbar"
            >
                {logViewItems}
            </div>
            if not logCollapsed
            else <span></span>}
        </div>;
}
