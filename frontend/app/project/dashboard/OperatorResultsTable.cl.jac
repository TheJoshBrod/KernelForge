"""Operator results table component."""

import from react { useState }
import from ..shared.MetricsAdapter { list_len, to_list }

def:pub OperatorResultsTable(props: any) -> Any {
    operatorRows = props.operatorRows;
    name = props.name;
    navigate = props.navigate;
    onOpenActionModal = props.onOpenActionModal;
    onOpenBenchLog = props.onOpenBenchLog;
    loading = props.loading if props.loading else False;

    [sortColumn, setSortColumn] = useState("ratio");
    [sortDirection, setSortDirection] = useState("desc");

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

        if sortColumn == "name" {
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
        ["w-12", "w-12", "w-12"]
    ];
    skeletonNodes = [];
    for i in range(3) {
        cells = [];
        for j in range(8) {
            bc = " border-l border-[var(--border)] pl-4" if j == 1 or j == 4 else "";
            cells.push(<td key={j} className={"py-3 px-4" + bc}><div className={"skeleton skeleton-line " + skeleton_widths[j][i]}></div></td>);
        }
        skeletonNodes.push(<tr key={i} className="border-b border-[var(--border)]">{cells}</tr>);
    }

    operatorRowsNodes = [];
    for r in sortedRows {
        if not r {
            continue;
        }
        r_stat = r["status"] if "status" in r else "Not started";
        r_short = r["short"] if "short" in r else "";
        r_baseline = r["baseline"] if "baseline" in r else "--";
        r_best = r["best"] if "best" in r else "--";
        r_speedup = r["speedup"] if "speedup" in r else "--";
        r_is_forged = r["is_forged"] if "is_forged" in r else False;
        r_name = r["name"] if "name" in r else "";

        is_benchmarked = r_stat == "Benchmarked";
        row_status_color = "text-zinc-500";
        if r_stat == "Forged ✓" { row_status_color = "text-emerald-400"; }
        elif is_benchmarked { row_status_color = "text-yellow-400"; }
        elif r_stat != "Not started" { row_status_color = "text-[var(--accent-orange)] animate-pulse"; }

        status_label = (r_stat + " —") if is_benchmarked else r_stat;
        speedup_color = "text-emerald-400 font-semibold" if r_is_forged else ("text-yellow-400" if is_benchmarked else "text-zinc-500");

        action_text = "Optimize" if r_stat == "Benchmarked" or r_stat == "Forged ✓" else "Generate";
        action_colors = "border-[var(--accent-orange)]/40 text-[var(--accent-orange)] hover:bg-[var(--accent-orange)] hover:text-zinc-950" if action_text == "Optimize" else "border-zinc-600 text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100 hover:border-zinc-500";
        action_btn_class = f"px-2 py-0.5 bg-transparent border rounded text-[10px] uppercase font-semibold transition-colors {action_colors}";

        r_runtime_pct = r["runtime_pct"] if "runtime_pct" in r else "--";
        r_call_count = r["call_count"] if "call_count" in r else "--";

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
                <td className="py-2 px-4 font-mono text-xs pr-4">
                    <button
                        data-opname={r_name}
                        data-action={action_text}
                        onClick={lambda e: any -> None { e.stopPropagation(); onOpenActionModal(e, e.currentTarget.getAttribute("data-opname"), e.currentTarget.getAttribute("data-action")); }}
                        className={action_btn_class}
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
                            <th className="py-2.5 px-4 font-normal text-right pr-4" title="Actions you can take on this operator">
                                <div className="flex items-center">Action</div>
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        {operatorRowsNodes if list_len(operatorRowsNodes) > 0 else (skeletonNodes if loading else [])}
                    </tbody>
                </table>
                {<div className="p-8 text-center text-xs text-zinc-500 font-mono">
                    No operators discovered yet. Profile the model to find operators.
                </div> if list_len(operatorRowsNodes) == 0 and not loading else <span></span>}
            </div>
        </div>;
}
