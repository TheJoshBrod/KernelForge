"""Workbench Log Panel Component"""

import from .....components.ui.Icons { ActivityIcon }

def:pub WorkbenchLog(props: any) -> Any {
    selectedNodeId = props.selectedNodeId;
    rawNodes = props.rawNodes;
    nodes = props.nodes;
    edges = props.edges;

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

        # Build formatted description
        desc_nodes = [];
        raw_desc = item["improvement_description"] if item["improvement_description"] else "";
        nl = String.fromCharCode(10);
        if raw_desc.startsWith("[FAILED]") {
            first_line = raw_desc.split(nl)[0];
            desc_nodes.push(<span key="err" className="text-red-400 text-[11px]">{first_line}</span>);
        } elif raw_desc {
            lines = raw_desc.split(nl);
            line_idx = 0;
            for line in lines {
                trimmed = line.trim();
                if trimmed {
                    if trimmed.startsWith("OPTIMIZATION:") or trimmed.startsWith("RATIONALE:") or trimmed.startsWith("IMPROVEMENT:") {
                        desc_nodes.push(<div key={line_idx} className="font-semibold text-zinc-300 text-[11px] mt-2 first:mt-0">{trimmed}</div>);
                    } elif trimmed.startsWith("- ") {
                        desc_nodes.push(
                            <div key={line_idx} className="flex gap-1.5 text-[11px] text-[var(--text-muted)] pl-2">
                                <span className="text-zinc-600 flex-shrink-0 mt-px">·</span>
                                <span>{trimmed.slice(2)}</span>
                            </div>
                        );
                    } else {
                        desc_nodes.push(<div key={line_idx} className="text-[11px] text-[var(--text-muted)]">{trimmed}</div>);
                    }
                }
                line_idx += 1;
            }
        }

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
                <div className="flex justify-between items-center mb-1.5">
                    <span className={"font-mono text-sm font-semibold " + ("text-[var(--accent)]" if f"{item['id']}" == selectedNodeId else "text-[var(--text-light)]")}>
                        Kernel{" " + item["id"]}
                    </span>
                    <div className={"font-mono text-sm font-semibold " + ("text-[var(--accent)]" if f"{item['id']}" == selectedNodeId else "text-[var(--text-light)]")}>
                        <span>{Number(item["value"] or 0).toFixed(4)}</span>
                    </div>
                </div>
                <div className="space-y-0.5">
                    {desc_nodes}
                </div>
            </div>
        );
        idx += 1;
    }

    return
        <div className="flex-1 flex flex-col min-h-0">
            <div className="h-10 bg-zinc-900/50 border-b border-zinc-800 flex items-center px-4 justify-between flex-shrink-0">
                <span className="text-xs font-semibold text-zinc-500 uppercase tracking-wider flex items-center gap-2">
                    <ActivityIcon className="w-4 h-4 text-[var(--accent)]"/>
                    Iteration Log
                </span>
                <span className="font-mono text-xs font-semibold text-zinc-400">
                    {"Kernel " + selectedNodeId if selectedNodeId else ""}
                </span>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-2 flex flex-col-reverse custom-scrollbar">
                {logViewItems}
            </div>
        </div>;
}
