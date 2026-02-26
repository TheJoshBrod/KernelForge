"""Operator results table component."""

import from ..shared.MetricsAdapter { list_len }

def:pub OperatorResultsTable(props: any) -> Any {
    operatorRows = props.operatorRows;
    name = props.name;
    navigate = props.navigate;
    onOpenActionModal = props.onOpenActionModal;
    onOpenBenchLog = props.onOpenBenchLog;
    loading = props.loading if props.loading else False;

    skeleton_widths = [["w-32", "w-40", "w-28"], ["w-16", "w-16", "w-16"], ["w-16", "w-16", "w-16"], ["w-12", "w-12", "w-12"], ["w-20", "w-20", "w-20"]];
    skeletonNodes = [];
    for i in range(3) {
        cells = [];
        for j in range(5) {
            cells.push(<td key={j} className="py-3 px-4"><div className={"skeleton skeleton-line " + skeleton_widths[j][i]}></div></td>);
        }
        skeletonNodes.push(<tr key={i} className="border-b border-[var(--border)]">{cells}</tr>);
    }

    operatorRowsNodes = [];
    for r in operatorRows {
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

        operatorRowsNodes.push(
            <tr
                key={r_name}
                className="border-b border-[var(--border)] bg-[var(--surface-1)] hover:bg-[var(--surface-2)] transition-colors cursor-pointer"
                onClick={lambda -> None { navigate(f"/project/{name}/operator-workbench/{r_name}"); }}
            >
                <td className="py-2.5 px-4 font-mono text-xs text-zinc-200" title={r_name}>{r_short}</td>
                <td className="py-2.5 px-4 font-mono text-xs text-zinc-400">{r_baseline}</td>
                <td className="py-2.5 px-4 font-mono text-xs text-zinc-300">{r_best}</td>
                <td className={"py-2.5 px-4 font-mono text-xs " + speedup_color}>{r_speedup}</td>
                <td className="py-2 px-4 font-mono text-xs flex items-center justify-between">
                    <span className={row_status_color}>{status_label}</span>
                    <button
                        data-opname={r_name}
                        data-action={action_text}
                        onClick={lambda e: any -> None { onOpenActionModal(e, e.currentTarget.getAttribute("data-opname"), e.currentTarget.getAttribute("data-action")); }}
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
                <div className="flex items-center gap-3">
                    <button
                        onClick={onOpenBenchLog}
                        className="text-[10px] px-2 py-1 rounded border border-[var(--border)] bg-[var(--surface-1)] hover:bg-[var(--surface-2)] text-zinc-300 transition-colors"
                    >
                        View Log
                    </button>
                </div>
            </div>
            <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                    <thead>
                        <tr className="bg-zinc-900/80 border-b border-[var(--border)] text-[10px] uppercase tracking-wider text-zinc-500 font-semibold">
                            <th className="py-2.5 px-4 font-normal">Operator</th>
                            <th className="py-2.5 px-4 font-normal">PyTorch</th>
                            <th className="py-2.5 px-4 font-normal">Best Kernel</th>
                            <th className="py-2.5 px-4 font-normal">Speedup</th>
                            <th className="py-2.5 px-4 font-normal">Status</th>
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
