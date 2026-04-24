"""Operator results table component."""

import from react { useState }
import from ..shared.MetricsAdapter { list_len, to_list }

def _index_list(count: int) -> list {
    out = [];
    idx = 0;
    while idx < count {
        out.push(idx);
        idx = idx + 1;
    }
    return out;
}

def:pub OperatorResultsTable(props: any) -> Any {
    operatorRows = props.operatorRows;
    name = props.name;
    navigate = props.navigate;
    onOpenActionModal = props.onOpenActionModal;
    onOpenBenchLog = props.onOpenBenchLog;
    loading = props.loading if props.loading else False;
    profiling = props.profiling if props and "profiling" in props else False;
    gpuMismatch = props.gpuMismatch if props and "gpuMismatch" in props else False;
    llmUsage = props.llmUsage if props and "llmUsage" in props else None;

    llmUsageByOp = {};
    if llmUsage and "per_operator" in llmUsage {
        per_op_list = to_list(llmUsage["per_operator"]);
        for entry in per_op_list {
            if entry and "operator" in entry {
                llmUsageByOp[entry["operator"]] = entry;
            }
        }
    }

    [sortColumn, setSortColumn] = useState("display_order");
    [sortDirection, setSortDirection] = useState("asc");

    def handleSort(column: str) -> None {
        if sortColumn == column {
            setSortDirection("asc" if sortDirection == "desc" else "desc");
        } else {
            setSortColumn(column);
            setSortDirection("desc");
        }
    }

    def getSortIcon(column: str) -> Any {
        if sortColumn != column {
            return <span className="opacity-0 group-hover:opacity-40 ml-1 transition-opacity">↕</span>;
        }
        return <span className="text-zinc-300 ml-1">{ "↓" if sortDirection == "desc" else "↑" }</span>;
    }

    # Sort the rows before rendering
    sortedRows = to_list(operatorRows);
    sortedRows.sort(lambda a: dict, b: dict -> int {
        if not a and not b { return 0; }
        if not a { return 1; }
        if not b { return -1; }

        valA = None;
        valB = None;

        if sortColumn == "display_order" {
            valA = a["display_order"] if "display_order" in a else 0;
            valB = b["display_order"] if "display_order" in b else 0;
        } elif sortColumn == "name" {
            valA = a["short"] if "short" in a else "";
            valB = b["short"] if "short" in b else "";
        } elif sortColumn == "baseline" {
            valA = a["baseline_raw"] if "baseline_raw" in a else 0.0;
            valB = b["baseline_raw"] if "baseline_raw" in b else 0.0;
        } elif sortColumn == "runtime_pct" {
            # Parse runtime_pct string to number for sorting ("<0.1%" -> 0, "12.5%" -> 12.5)
            pctA = a["runtime_pct"] if "runtime_pct" in a else "0";
            pctB = b["runtime_pct"] if "runtime_pct" in b else "0";
            valA = 0.0 if pctA == "<0.1%" or pctA == "--" else parseFloat(pctA.replace("%", ""));
            valB = 0.0 if pctB == "<0.1%" or pctB == "--" else parseFloat(pctB.replace("%", ""));
        } elif sortColumn == "call_count" {
            valA = a["call_count"] if "call_count" in a else 0;
            valB = b["call_count"] if "call_count" in b else 0;
        } elif sortColumn == "best" {
            # In Dashboard, we didn't pass "best_raw" to OperatorResultsTable. We can approximate or just parse the string.
            # Wait, Dashboard already puts `ratio` in the rows. Let's just parse the best string if it exists? 
            # Actually, baseline_raw exists. If we have ratio, we can sort by ratio for speedup.
            strA = a["best"] if "best" in a else "--";
            strB = b["best"] if "best" in b else "--";
            valA = 999999.0 if strA == "--" else parseFloat(strA.replace("ms", "").trim());
            valB = 999999.0 if strB == "--" else parseFloat(strB.replace("ms", "").trim());
        } elif sortColumn == "speedup" or sortColumn == "ratio" {
            valA = a["ratio"] if "ratio" in a else 0.0;
            valB = b["ratio"] if "ratio" in b else 0.0;
        } elif sortColumn == "status" {
            valA = a["status"] if "status" in a else "";
            valB = b["status"] if "status" in b else "";
        }

        # Handle None
        if valA == None and valB == None { return 0; }
        if valA == None { return 1 if sortDirection == "desc" else -1; }
        if valB == None { return -1 if sortDirection == "desc" else 1; }

        if valA < valB {
            return 1 if sortDirection == "desc" else -1;
        }
        if valA > valB {
            return -1 if sortDirection == "desc" else 1;
        }
        return 0;
    });

    skeleton_widths = [
        ["w-32", "w-40", "w-28"],
        ["w-16", "w-16", "w-16"],
        ["w-12", "w-12", "w-12"],
        ["w-8", "w-8", "w-8"],
        ["w-16", "w-16", "w-16"],
        ["w-12", "w-12", "w-12"],
        ["w-20", "w-20", "w-20"],
        ["w-16", "w-16", "w-16"],
        ["w-12", "w-12", "w-12"],
        ["w-12", "w-12", "w-12"]
    ];
    skeletonNodes = [];
    row_indices = _index_list(3);
    col_indices = _index_list(10);
    for i in row_indices {
        cells = [];
        for j in col_indices {
            bc = " border-l border-[var(--border)] pl-4" if j == 1 or j == 4 or j == 8 else "";
            cells.push(<td key={j} className={"py-3 px-4" + bc}><div className={"skeleton skeleton-line " + skeleton_widths[j][i]}></div></td>);
        }
        skeletonNodes.push(<tr key={i} className="border-b border-[var(--border)]">{cells}</tr>);
    }

    operatorRowsNodes = [];
    for r in sortedRows {
        if not r {
            continue;
        }
        raw_r_stat = r["status"] if "status" in r else "Not started";
        r_stat = "Benchmarked" if raw_r_stat == "Done" else raw_r_stat;
        r_short = r["short"] if "short" in r else "";
        r_baseline = r["baseline"] if "baseline" in r else "--";
        r_best = r["best"] if "best" in r else "--";
        r_speedup = r["speedup"] if "speedup" in r else "--";
        r_is_forged = r["is_forged"] if "is_forged" in r else False;
        r_name = r["name"] if "name" in r else "";
        r_can_optimize_action = bool(r["can_optimize_action"]) if "can_optimize_action" in r and r["can_optimize_action"] else False;
        r_action_disabled = bool(r["action_disabled"]) if "action_disabled" in r and r["action_disabled"] else False;

        is_benchmarked = r_stat == "Benchmarked";
        row_status_color = "text-zinc-500";
        if r_stat == "Failed" { row_status_color = "text-red-400"; }
        elif r_stat == "Forged ✓" { row_status_color = "text-emerald-400"; }
        elif is_benchmarked { row_status_color = "text-yellow-400"; }
        elif r_stat != "Not started" { row_status_color = "text-[var(--accent-orange)] animate-pulse"; }

        status_label = (r_stat + ":") if is_benchmarked else r_stat;
        speedup_color = "text-emerald-400 font-semibold" if r_is_forged else ("text-yellow-400" if is_benchmarked else "text-zinc-500");

        action_text = "Optimize" if r_can_optimize_action else "Generate";
        action_colors = "border-[var(--accent-orange)]/40 text-[var(--accent-orange)] hover:bg-[var(--accent-orange)] hover:text-zinc-950" if action_text == "Optimize" else "border-zinc-600 text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100 hover:border-zinc-500";
        action_disabled = gpuMismatch or r_action_disabled;
        disabled_btn_class = " opacity-40 cursor-not-allowed pointer-events-none hover:bg-transparent hover:text-current hover:border-current";
        gpu_disabled_class = disabled_btn_class if action_disabled else "";
        action_btn_class = f"px-2 py-0.5 bg-transparent border rounded text-[10px] uppercase font-semibold transition-colors {action_colors}{gpu_disabled_class}";

        r_runtime_pct = r["runtime_pct"] if "runtime_pct" in r else "--";
        r_call_count = r["call_count"] if "call_count" in r else "--";

        llm_cell_tokens = "--";
        llm_cell_cost = "--";
        llm_entry = llmUsageByOp[r_name] if r_name in llmUsageByOp else None;
        if llm_entry {
            total_toks = (llm_entry["input_tokens"] if "input_tokens" in llm_entry else 0)
                + (llm_entry["output_tokens"] if "output_tokens" in llm_entry else 0);
            llm_cell_tokens = total_toks.toLocaleString();
            cost_val = llm_entry["total_cost_usd"] if "total_cost_usd" in llm_entry else 0.0;
            llm_cell_cost = "$" + cost_val.toFixed(4);
        }

        operatorRowsNodes.push(
            <tr
                key={r_name}
                className="border-b border-[var(--border)] bg-[var(--surface-1)] hover:bg-[var(--surface-2)] transition-colors cursor-pointer group"
                onClick={lambda -> None { navigate(f"/project/{name}/operator-workbench/{r_name}"); }}
            >
                <td className="py-2.5 px-4 font-mono text-xs text-zinc-200" title={r_name}>{r_short}</td>
                
                <td className="py-2.5 px-4 font-mono text-xs text-zinc-400 border-l border-[var(--border)] pl-4 transition-colors">{r_baseline}</td>
                <td className="py-2.5 px-4 font-mono text-xs text-zinc-400">{r_runtime_pct}</td>
                <td className="py-2.5 px-4 font-mono text-xs text-zinc-500 pr-4">{r_call_count}</td>
                
                <td className="py-2.5 px-4 font-mono text-xs text-zinc-300 border-l border-[var(--border)] pl-4 transition-colors">{r_best}</td>
                <td className={"py-2.5 px-4 font-mono text-xs " + speedup_color}>{r_speedup}</td>
                <td className="py-2.5 px-4 font-mono text-xs">
                    <span className={row_status_color}>{status_label}</span>
                </td>
                <td className="py-2.5 px-4 font-mono text-xs text-zinc-400 border-l border-[var(--border)] pl-4" title="Total tokens consumed for this operator">{llm_cell_tokens}</td>
                <td className="py-2.5 px-4 font-mono text-xs text-emerald-300/80 pr-4" title="Total USD cost for this operator">{llm_cell_cost}</td>
                <td className="py-2 px-4 font-mono text-xs pr-4" onClick={lambda e: any -> None { e.stopPropagation(); }}>
                    <button
                        data-opname={r_name}
                        data-action={action_text}
                        onClick={lambda e: any -> None {
                            e.stopPropagation();
                            if action_disabled {
                                return;
                            }
                            onOpenActionModal(
                                e,
                                e.currentTarget.getAttribute("data-opname"),
                                e.currentTarget.getAttribute("data-action")
                            );
                        }}
                        className={action_btn_class}
                        disabled={action_disabled}
                        title={
                            "This operator already has queued or active work."
                            if r_action_disabled
                            else ("Current GPU does not match the benchmarked GPU." if gpuMismatch else action_text)
                        }
                    >
                        {action_text}
                    </button>
                </td>
            </tr>
        );
    }

    return
        <div className="rounded-md border border-[var(--border)] bg-[var(--surface-0)] overflow-hidden">
            <div className="p-3 border-b border-[var(--border)] bg-zinc-900/40 flex justify-between items-center">
                <h3 className="text-xs font-semibold uppercase tracking-[0.12em] text-zinc-200">
                    Operator Results
                </h3>
            </div>
            <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                    <thead>
                        <tr className="bg-zinc-900/80 border-b border-[var(--border)] text-[10px] uppercase tracking-wider text-zinc-500 font-semibold select-none">
                            <th className="py-2.5 px-4 font-normal cursor-pointer hover:bg-zinc-800/50 transition-colors group" title="Operator name" onClick={lambda -> None { handleSort("name"); }}>
                                <div className="flex items-center">Operator {getSortIcon("name")}</div>
                            </th>
                            
                            <th className="py-2.5 px-4 font-normal border-l border-[var(--border)] pl-4 cursor-pointer hover:bg-zinc-800/50 transition-colors group" title="PyTorch baseline execution time" onClick={lambda -> None { handleSort("baseline"); }}>
                                <div className="flex items-center">PyTorch Time {getSortIcon("baseline")}</div>
                            </th>
                            <th className="py-2.5 px-4 font-normal cursor-pointer hover:bg-zinc-800/50 transition-colors group" title="Percentage of total PyTorch execution time" onClick={lambda -> None { handleSort("runtime_pct"); }}>
                                <div className="flex items-center">% Run Time {getSortIcon("runtime_pct")}</div>
                            </th>
                            <th className="py-2.5 px-4 font-normal pr-4 cursor-pointer hover:bg-zinc-800/50 transition-colors group" title="Number of calls in model" onClick={lambda -> None { handleSort("call_count"); }}>
                                <div className="flex items-center">Calls {getSortIcon("call_count")}</div>
                            </th>

                            <th className="py-2.5 px-4 font-normal border-l border-[var(--border)] pl-4 cursor-pointer hover:bg-zinc-800/50 transition-colors group" title="Execution time of the best generated kernel" onClick={lambda -> None { handleSort("best"); }}>
                                <div className="flex items-center">Best Kernel {getSortIcon("best")}</div>
                            </th>
                            <th className="py-2.5 px-4 font-normal cursor-pointer hover:bg-zinc-800/50 transition-colors group" title="Speedup ratio over PyTorch baseline" onClick={lambda -> None { handleSort("ratio"); }}>
                                <div className="flex items-center">Speedup {getSortIcon("ratio")}</div>
                            </th>
                            <th className="py-2.5 px-4 font-normal cursor-pointer hover:bg-zinc-800/50 transition-colors group" title="Current progress of operator generation" onClick={lambda -> None { handleSort("status"); }}>
                                <div className="flex items-center">Status {getSortIcon("status")}</div>
                            </th>
                            <th className="py-2.5 px-4 font-normal border-l border-[var(--border)] pl-4" title="Total LLM tokens used for this operator">
                                <div className="flex items-center">Tokens</div>
                            </th>
                            <th className="py-2.5 px-4 font-normal pr-4" title="Total USD cost of LLM calls for this operator">
                                <div className="flex items-center">Cost</div>
                            </th>
                            <th className="py-2.5 px-4 font-normal text-right pr-4" title="Actions you can take on this operator">
                                <div className="flex items-center">Action</div>
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        {operatorRowsNodes if list_len(operatorRowsNodes) > 0 else (skeletonNodes if loading else [])}
                    </tbody>
                </table>
                {<div className="p-8 text-center text-xs text-zinc-500 font-mono animate-pulse">
                    Profiling in progress...
                </div> if list_len(operatorRowsNodes) == 0 and not loading and profiling else <span></span>}
                {<div className="p-8 text-center text-xs text-zinc-500 font-mono">
                    No operators discovered yet. Profile the model to find operators.
                </div> if list_len(operatorRowsNodes) == 0 and not loading and not profiling else <span></span>}
            </div>
        </div>;
}
