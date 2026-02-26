"""Work queue panel component."""

import from ..shared.MetricsAdapter { to_list, list_len }

def:pub WorkQueuePanel(props: any) -> Any {
    name = props.name;
    activeTasks = props.activeTasks;
    activePoll = props.activePoll;
    currentOperator = props.currentOperator;
    benchmarkSlot = props.benchmarkSlot;
    pendingOperators = props.pendingOperators;
    onClearQueue = props.onClearQueue;

    active_tasks_nodes = [];
    for task in to_list(activeTasks) {
        c_step = task["current_step"];
        tag = task["tag"];

        track_nodes = [];
        steps = ["Monitoring", "Generating", "Verifying", "Benchmarking"] if tag == "[GEN]" else ["Generating", "Validating", "Profiling"];

        c_idx = -1;
        for i in range(list_len(steps)) {
            if steps[i] == c_step {
                c_idx = i;
            }
        }

        for i in range(list_len(steps)) {
            s = steps[i];
            is_active = (i == c_idx);
            is_done = (c_step == "Done" or c_step == "Failed" or (c_idx > -1 and i < c_idx));

            s_label = s;
            if is_active {
                s_label = f"[{s}]";
                track_nodes.push(<span key={s} className="text-[var(--accent-orange)] font-semibold">{s_label}</span>);
            } elif is_done {
                s_label = s + (" ✓" if c_step != "Failed" else "");
                track_nodes.push(<span key={s} className="text-zinc-500">{s_label}</span>);
            } else {
                track_nodes.push(<span key={s} className="text-zinc-600">{s_label}</span>);
            }
            if i < list_len(steps) - 1 {
                track_nodes.push(<span key={s + "sep"} className="text-zinc-700 mx-1.5">→</span>);
            }
        }

        status_label = c_step;
        if c_step == "Done" {
            status_label = "Done " + (task["result"] if "result" in task else "");
        } elif c_step == "Failed" {
            status_label = "Failed";
        }

        bar_w = "0%";
        if c_step == "Generating" { bar_w = "33%"; }
        elif c_step == "Verifying" { bar_w = "66%"; }
        elif c_step == "Validating" { bar_w = "66%"; }
        elif c_step == "Benchmarking" { bar_w = "85%"; }
        elif c_step == "Profiling" { bar_w = "85%"; }
        elif c_step == "Done" { bar_w = "100%"; }

        sub_info = "";
        if "iter_current" in task and "iter_max" in task {
            sub_info = f"iter {str(task['iter_current'])}/{str(task['iter_max'])} · ";
        }
        if tag == "[OPT]" and "parent_ref" in task and task["parent_ref"] {
            sub_info = sub_info + "parent: " + task["parent_ref"] + " · ";
        }
        if c_step == "Failed" {
            sub_info = sub_info + (task["result"] if "result" in task else "validation failed");
        } else {
            atmpt = str(task["attempt_current"] if "attempt_current" in task else 1);
            max_atmpt = str(task["attempt_max"] if "attempt_max" in task else 3);
            sub_info = sub_info + f"attempt {atmpt}/{max_atmpt}";
        }

        active_tasks_nodes.push(
            <div key={task["id"]} className="p-4 bg-[var(--surface-0)] hover:bg-[var(--surface-1)] transition-colors group">
                <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3 font-mono text-xs">
                        <span className={"font-semibold " + ("text-emerald-400" if tag == "[GEN]" else "text-blue-400")}>{tag}</span>
                        <span className="text-zinc-200">{task["op_name"]} · node {task["id"]}</span>
                    </div>
                    <div className="flex items-center gap-3">
                        {<div className="w-24 h-1.5 bg-zinc-900 rounded overflow-hidden">
                            <div className={"h-full transition-all duration-500 " + ("bg-emerald-500" if c_step == "Done" else ("bg-red-500" if c_step == "Failed" else "forge-progress-fill"))} style={{"width": bar_w}}></div>
                        </div> if c_step != "Failed" and c_step != "Done" else <span></span>}
                        <span className={"font-mono text-xs text-right " + ("text-red-400" if c_step == "Failed" else ("text-emerald-400" if c_step == "Done" else "text-zinc-400"))}>{status_label}</span>
                    </div>
                </div>
                <div className="flex items-center justify-between font-mono text-[10px] text-zinc-500 pl-11">
                    <div className="flex items-center">
                        {track_nodes}
                    </div>
                    <div className="text-zinc-600 truncate max-w-[200px]" title={sub_info}>
                        {sub_info}
                    </div>
                </div>
            </div>
        );
    }

    return
        <div className="rounded-md border border-[var(--border)] bg-[var(--surface-0)] overflow-hidden">
            <div className="p-3 border-b border-[var(--border)] bg-zinc-900/40 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="flex items-center gap-3 shrink-0">
                    <h3 className="text-xs font-semibold uppercase tracking-[0.12em] text-zinc-200">
                        Work Queue {("(" + currentOperator + ")") if currentOperator else ""}
                    </h3>
                    <button
                        onClick={onClearQueue}
                        className="text-[10px] px-2 py-0.5 rounded border border-zinc-700 bg-zinc-900 hover:border-zinc-500 hover:text-zinc-200 text-zinc-500 transition-colors font-mono"
                    >
                        clear
                    </button>
                </div>
                {<div className="flex flex-wrap items-center gap-2 sm:gap-3 text-xs font-mono bg-black/20 rounded px-3 py-1.5 border border-[var(--border)]">
                    <div className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-orange)] animate-pulse"></span>
                        <span className="text-zinc-500 uppercase tracking-wider text-[10px]">GPU Slot:</span>
                    </div>
                    <span className="text-zinc-200 font-semibold">{benchmarkSlot["now"] if benchmarkSlot["now"] else "idle"}</span>
                    {<span className="text-zinc-500">· Next: <span className="text-zinc-400">{", ".join(pendingOperators)}</span></span> if list_len(pendingOperators) > 0 else <span></span>}
                </div> if activePoll else <span></span>}
            </div>
            <div className="divide-y divide-[var(--border)] max-h-[400px] overflow-y-auto">
                {active_tasks_nodes
                if list_len(active_tasks_nodes) > 0 and activePoll
                else <div className="p-8 text-center text-xs text-zinc-500 font-mono">Queue empty</div>}
            </div>
        </div>;
}
