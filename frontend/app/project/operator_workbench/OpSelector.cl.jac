"""Operator Selector Component"""

import from react { useState, useEffect }
import from "react-router-dom" { useNavigate, useParams }

def:pub OpSelector -> Any {
    has projectName: str;

    navigate = useNavigate();
    params = useParams();
    resolvedProjectName = projectName or params.projectName;

    raw_ops_state = useState([]);
    rawOps = raw_ops_state[0];
    setRawOps = raw_ops_state[1];

    loading_ops_state = useState(True);
    loadingOps = loading_ops_state[0];
    setLoadingOps = loading_ops_state[1];

    filter_state = useState("");
    filterText = filter_state[0];
    setFilterText = filter_state[1];

    sort_state = useState("count");
    sortMode = sort_state[0];
    setSortMode = sort_state[1];

    def format_op(raw: str) -> str {
        raw_text = raw or "";
        prefix = "torch_nn_functional_";
        if raw_text.startsWith(prefix) {
            return raw_text.replace(prefix, "");
        }
        return raw_text;
    }

    def to_list(items: Any) -> list {
        if not items {
            return [];
        }
        try {
            if Array.isArray(items) {
                return items;
            }
        } except Exception {
        # no-op
        }
        normalized = [];
        try {
            for item in items {
                normalized.push(item);
            }
            return normalized;
        } except Exception {
        # no-op
        }
        try {
            return Object.values(items);
        } except Exception {
            return [];
        }
    }

    def count_desc(a: Any, b: Any) -> int {
        ca = a["count"] if a else 0;
        cb = b["count"] if b else 0;
        return cb - ca;
    }

    # Stateless comparator — sort key is pre-stamped onto each item as _sort_val
    def sort_val_desc(a: Any, b: Any) -> int {
        va = a["_sort_val"] if a and "_sort_val" in a else 0;
        vb = b["_sort_val"] if b and "_sort_val" in b else 0;
        return vb - va;
    }

    async def fetchOps -> None {
        if not resolvedProjectName {
            console.error("No project name found in OpSelector");
            return;
        }
        setLoadingOps(True);
        try {
            ops_resp = await __jacSpawn(
                "GetProjectOps", "", {"projectName": resolvedProjectName}
            );
            ops_data = ops_resp.reports[0] if ops_resp and ops_resp.reports else None;
            raw_list = to_list(
                ops_data["ops"] if ops_data and "ops" in ops_data else []
            );
            op_list = [];
            for item in raw_list {
                if item { op_list.push(item); }
            }

            # Compute total inference time across all ops for % calculation
            total_time = 0.0;
            for item in op_list {
                ms = item["profile_ms"] if "profile_ms" in item else 0.0;
                cnt = (item["count"] * 1) if "count" in item and item["count"] else 1;
                total_time = total_time + (ms * cnt);
            }

            enriched = [];
            for item in op_list {
                ms = item["profile_ms"] if "profile_ms" in item else 0.0;
                cnt = (item["count"] * 1) if "count" in item and item["count"] else 1;
                total_ms = ms * cnt;
                pct = (total_ms / total_time * 100.0) if total_time > 0.0 else 0.0;
                enriched.push({**item, "total_time_ms": total_ms, "pct_time": pct, "avg_ms": ms});
            }
            enriched.sort(count_desc);
            setRawOps(enriched);
        } except Exception as e {
            console.error("Failed to fetch ops:", e);
            setRawOps([]);
        } finally {
            setLoadingOps(False);
        }
    }

    useEffect(lambda -> None { fetchOps();}, [resolvedProjectName]);

    # Stamp _sort_val onto each copy so the comparator needs no closure over sortMode
    display_ops = [];
    for o in rawOps {
        sv = o["count"] if "count" in o else 0;
        if sortMode == "pct_time" {
            sv = o["pct_time"] if "pct_time" in o else 0.0;
        }
        if sortMode == "avg_ms" {
            sv = o["avg_ms"] if "avg_ms" in o else 0.0;
        }
        display_ops.push({**o, "_sort_val": sv});
    }
    display_ops.sort(sort_val_desc);

    sort_options = [
        {"key": "count", "label": "Frequency"},
        {"key": "pct_time", "label": "% Time"},
        {"key": "avg_ms", "label": "Avg ms"},
    ];
    sort_btns = [];
    for opt in sort_options {
        is_active = sortMode == opt["key"];
        btn_class = (
            "px-3 py-1.5 text-xs font-medium rounded border transition-colors cursor-pointer " +
            ("bg-[var(--accent)] border-[var(--accent)] text-[var(--bg-darker)]"
             if is_active
             else "bg-transparent border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--accent)] hover:text-[var(--text-light)]")
        );
        opt_key = opt["key"];
        sort_btns.push(
            <button
                key={opt["key"]}
                className={btn_class}
                onClick={lambda -> None { setSortMode(opt_key); }}
            >
                {opt["label"]}
            </button>
        );
    }

    op_items = [];
    for op in display_ops {
        if not format_op(op["name"]).toLowerCase().includes(filterText.toLowerCase()) {
            continue;
        }
        badge_text = str(op["count"]) + " calls";
        if sortMode == "pct_time" and "pct_time" in op {
            badge_text = op["pct_time"].toFixed(1) + "% time";
        }
        if sortMode == "avg_ms" and "avg_ms" in op {
            badge_text = op["avg_ms"].toFixed(2) + " ms avg";
        }
        op_items.push(
            <div
                key={op["name"]}
                className="clip-anvil p-4 bg-[var(--bg-panel)] border border-[var(--border)] hover:border-[var(--accent)] hover:shadow-[inset_4px_0_0_var(--accent)] hover:!border-l-2 hover:!border-l-[var(--accent)] cursor-pointer transition-all flex justify-between items-center group"
                onClick={lambda -> None { navigate(
                    f"/project/{resolvedProjectName}/operator-workbench/{op['name']}"
                );}}
            >
                <span className="font-mono text-sm text-[var(--text-light)] group-hover:text-white">
                    {format_op(op["name"])}
                </span>
                <span className="text-xs text-[var(--text-muted)] bg-[var(--bg-darker)] px-2 py-1 rounded">
                    {badge_text}
                </span>
            </div>
        );
    }

    return
        <div className="flex-1 flex flex-col p-8 bg-[var(--bg-charcoal)] text-[var(--text-light)] h-screen overflow-hidden">
            <div className="mb-6">
                <h2 className="text-2xl font-bold text-white mb-2">
                    Operator Workbench
                </h2>
                <p className="text-[var(--text-muted)]">
                    Select an operator to explore.
                </p>
            </div>
            <div className="flex items-center gap-4 mb-4">
                <input
                    type="text"
                    placeholder="Filter operators..."
                    value={filterText}
                    onChange={lambda e: any -> None { setFilterText(e.target.value); }}
                    className="flex-1 max-w-sm px-3 py-2 bg-[var(--bg-panel)] border border-[var(--border)] text-[var(--text-light)] placeholder-[var(--text-muted)] text-sm focus:outline-none focus:border-[var(--accent)] rounded"
                />
                <div className="flex items-center gap-2 ml-auto">
                    <span className="text-xs text-[var(--text-muted)]">{"Sort:"}</span>
                    {sort_btns}
                </div>
            </div>
            <hr className="border-[var(--border)] mb-4" />
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 overflow-y-auto pb-20">
                {op_items}{<div className="col-span-full text-center text-[var(--text-muted)] py-12 italic">
                    Loading operators...
                </div>
                if loadingOps
                else <div className="col-span-full text-center text-[var(--text-muted)] py-12 italic">
                    No operators found.
                </div>
                if rawOps.length == 0
                else <span></span>}
            </div>
        </div>;
}
