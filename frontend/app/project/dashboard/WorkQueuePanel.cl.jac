"""Work queue panel component."""

import from react { useState, useEffect }
import from ..shared.MetricsAdapter { to_list, list_len, op_matches, to_num }

def:pub WorkQueuePanel(props: any) -> Any {
    name = props.name;
    activeTasks = props.activeTasks;
    activePoll = props.activePoll;
    currentOperator = props.currentOperator;
    benchmarkSlot = props.benchmarkSlot;
    pendingOperators = props.pendingOperators;
    onClearQueue = props.onClearQueue;
    isStarting = props.isStarting if "isStarting" in props else False;
    startingTag = props.startingTag if "startingTag" in props else "";
    startingOp = props.startingOp if "startingOp" in props else "";
    speedData = props.speedData if "speedData" in props else [];

    modal_state = useState(False);
    modal_open = modal_state[0];
    setModalOpen = modal_state[1];

    history_state = useState([]);
    completedHistory = history_state[0];
    setCompletedHistory = history_state[1];

    # Accumulate Done/Failed tasks from activeTasks into persistent history.
    # Update existing entries (e.g. when value_ms is added after Benchmarking).
    useEffect(lambda -> None {
        for task in to_list(activeTasks) {
            if task["current_step"] == "Done" or task["current_step"] == "Failed" {
                setCompletedHistory(lambda prev: any -> Any {
                    found = False;
                    next_list = [];
                    for t in to_list(prev) {
                        if str(t["id"]) == str(task["id"]) {
                            found = True;
                            next_list.push(task);
                        } else {
                            next_list.push(t);
                        }
                    }
                    if not found { next_list.push(task); }
                    return next_list;
                });
            }
        }
    }, [activeTasks]);

    # Active queue: only in-progress tasks from the live poll
    in_progress = [];
    for task in to_list(activeTasks) {
        c_step = task["current_step"];
        if c_step != "Done" and c_step != "Failed" {
            in_progress.push(task);
        }
    }

    # Completed section uses accumulated history, not the live poll
    completed = completedHistory;

    # ---- Active task nodes ----
    active_tasks_nodes = [];
    for task in in_progress {
        c_step = task["current_step"];
        tag = task["tag"];

        track_nodes = [];
        steps = ["Monitoring", "Generating", "Validating", "Benchmarking"] if tag == "[GEN]" else ["Generating", "Validating", "Profiling"];

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

            if is_active {
                track_nodes.push(<span key={s} className="text-[var(--accent-orange)] font-semibold">[{s}]</span>);
            } elif is_done {
                track_nodes.push(<span key={s} className="text-zinc-500">{s + (" ✓" if c_step != "Failed" else "")}</span>);
            } else {
                track_nodes.push(<span key={s} className="text-zinc-600">{s}</span>);
            }
            if i < list_len(steps) - 1 {
                track_nodes.push(<span key={s + "sep"} className="text-zinc-700 mx-1.5">→</span>);
            }
        }

        status_label = c_step;
        if c_step == "Failed" {
            status_label = "Failed";
        }

        bar_w = "0%";
        if c_step == "Generating" { bar_w = "33%"; }
        elif c_step == "Validating" { bar_w = "66%"; }
        elif c_step == "Benchmarking" { bar_w = "85%"; }
        elif c_step == "Profiling" { bar_w = "85%"; }
        elif c_step == "Done" { bar_w = "100%"; }

        sub_info = "";
        if "iter_current" in task and "iter_max" in task {
            sub_info = f"candidate {task['iter_current']}/{task['iter_max']} · ";
        }
        if tag == "[OPT]" and "parent_ref" in task and task["parent_ref"] {
            sub_info = sub_info + "parent: " + task["parent_ref"] + " · ";
        }
        if c_step == "Failed" {
            sub_info = sub_info + (task["result"] if "result" in task else "validation failed");
        } elif c_step == "Generating" or c_step == "Validating" {
            atmpt = str(task["attempt_current"] if "attempt_current" in task else 1);
            max_atmpt = str(task["attempt_max"] if "attempt_max" in task else 3);
            sub_info = sub_info + f"attempt {atmpt}/{max_atmpt}";
        } else {
            sub_info = sub_info + "queued";
        }

        active_tasks_nodes.push(
            <div key={task["id"]} className="p-4 bg-[var(--surface-0)] hover:bg-[var(--surface-1)] transition-colors">
                <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3 font-mono text-xs">
                        <span className={"font-semibold " + ("text-emerald-400" if tag == "[GEN]" else "text-blue-400")}>{tag}</span>
                        <span className="text-zinc-200">{task["op_name"]} · node {task["id"]}</span>
                    </div>
                    <div className="flex items-center gap-3">
                        {<div className="w-24 h-1.5 bg-zinc-900 rounded overflow-hidden">
                            <div className="h-full transition-all duration-500 forge-progress-fill" style={{"width": bar_w}}></div>
                        </div> if c_step != "Failed" and c_step != "Done" else <span></span>}
                        <span className={"font-mono text-xs text-right " + ("text-red-400" if c_step == "Failed" else "text-zinc-400")}>{status_label}</span>
                    </div>
                </div>
                <div className="flex flex-col gap-0.5 font-mono text-[10px] text-zinc-500 pl-11">
                    <div className="flex items-center">
                        {track_nodes}
                    </div>
                    <div className="text-zinc-600" title={sub_info}>
                        {sub_info}
                    </div>
                </div>
            </div>
        );
    }

    # ---- Completed section: group by op, show best of each batch ----
    op_group_keys = [];
    op_groups = {};
    for task in completed {
        op = str(task["op_name"]);
        if not (op in op_groups) {
            op_group_keys.push(op);
            op_groups[op] = {"best_task": None, "best_ms": None, "any_done": False, "tag": task["tag"]};
        }
        if task["current_step"] == "Done" {
            op_groups[op]["any_done"] = True;
            v = to_num(task["value_ms"] if "value_ms" in task else None);
            if v != None {
                if op_groups[op]["best_ms"] == None or v < op_groups[op]["best_ms"] {
                    op_groups[op]["best_ms"] = v;
                    op_groups[op]["best_task"] = task;
                }
            }
        }
    }

    completed_summary_rows = [];
    for op_key in op_group_keys {
        grp = op_groups[op_key];
        best_task = grp["best_task"];
        best_ms = grp["best_ms"];
        any_done = grp["any_done"];
        tag = grp["tag"];

        # Get pytorch_ms for this op
        pytorch_ms = None;
        for sd in to_list(speedData) {
            if sd and "name" in sd and op_matches(str(sd["name"]), op_key) {
                pytorch_ms = to_num(sd["pytorch"] if "pytorch" in sd else None);
            }
        }

        color = "red";
        if any_done {
            if best_ms != None and pytorch_ms != None and best_ms > 0 and pytorch_ms > 0 and best_ms < pytorch_ms {
                color = "green";
            } else {
                color = "yellow";
            }
        }

        dot_cls = "bg-red-500" if color == "red" else ("bg-emerald-500" if color == "green" else "bg-yellow-400");
        text_cls = "text-red-400" if color == "red" else ("text-emerald-400" if color == "green" else "text-yellow-400");
        border_cls = "border-l-2 border-red-500/40" if color == "red" else ("border-l-2 border-emerald-500/40" if color == "green" else "border-l-2 border-yellow-400/40");
        tag_cls = "text-emerald-400" if tag == "[GEN]" else "text-blue-400";

        perf_label = "all failed";
        if any_done {
            perf_label = "done";
        }
        if best_task != None and best_ms != None {
            perf_label = f"Kernel {best_task['id']} · {Number(best_ms).toFixed(4)} ms";
        }

        completed_summary_rows.push(
            <div key={op_key} className={"px-4 py-2.5 flex items-center justify-between " + border_cls}>
                <div className="flex items-center gap-2.5 font-mono text-xs min-w-0">
                    <span className={"w-1.5 h-1.5 rounded-full flex-shrink-0 " + dot_cls}></span>
                    <span className={"font-semibold flex-shrink-0 " + tag_cls}>{tag}</span>
                    <span className="text-zinc-300 truncate">{op_key}</span>
                </div>
                <span className={"font-mono text-xs flex-shrink-0 ml-3 " + text_cls}>{perf_label}</span>
            </div>
        );
    }

    # ---- View completed jobs modal: all individual tasks ----
    modal_list_rows = [];
    for task in completed {
        mt_op = str(task["op_name"]);
        mt_step = task["current_step"];
        mt_id = task["id"];
        mt_tag = task["tag"];
        mt_v = to_num(task["value_ms"] if "value_ms" in task else None);

        # Get pytorch_ms for color
        mt_pytorch = None;
        for sd in to_list(speedData) {
            if sd and "name" in sd and op_matches(str(sd["name"]), mt_op) {
                mt_pytorch = to_num(sd["pytorch"] if "pytorch" in sd else None);
            }
        }

        mt_color = "yellow";
        if mt_step == "Failed" {
            mt_color = "red";
        } elif mt_v != None and mt_pytorch != None and mt_v > 0 and mt_pytorch > 0 and mt_v < mt_pytorch {
            mt_color = "green";
        }

        mt_dot = "bg-red-500" if mt_color == "red" else ("bg-emerald-500" if mt_color == "green" else "bg-yellow-400");
        mt_text = "text-red-400" if mt_color == "red" else ("text-emerald-400" if mt_color == "green" else "text-yellow-400");
        mt_tag_cls = "text-emerald-400" if mt_tag == "[GEN]" else "text-blue-400";

        perf_str = "failed";
        if mt_step == "Done" {
            perf_str = f"{Number(mt_v).toFixed(4)} ms" if mt_v != None else "done";
        }

        modal_list_rows.push(
            <div key={mt_id} className="flex items-center justify-between px-4 py-2.5 border-b border-zinc-800/60 last:border-0">
                <div className="flex items-center gap-2.5 font-mono text-xs min-w-0">
                    <span className={"w-1.5 h-1.5 rounded-full flex-shrink-0 " + mt_dot}></span>
                    <span className={"font-semibold flex-shrink-0 " + mt_tag_cls}>{mt_tag}</span>
                    <span className="text-zinc-300 truncate">{mt_op} · kernel {mt_id}</span>
                </div>
                <span className={"font-mono text-xs flex-shrink-0 ml-3 " + mt_text}>{perf_str}</span>
            </div>
        );
    }

    modal_el = <span/>;
    if modal_open {
        completed_count_label = f"{completed.length} task" + ("s" if completed.length != 1 else "");
        modal_el =
            <div
                className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
                onClick={lambda -> None { setModalOpen(False); }}
            >
                <div
                    className="w-full max-w-lg mx-4 bg-[var(--surface-0)] border border-[var(--border)] rounded-xl shadow-2xl overflow-hidden flex flex-col max-h-[70vh]"
                    onClick={lambda e: any -> None { e.stopPropagation(); }}
                >
                    <div className="px-5 py-4 border-b border-[var(--border)] flex items-center justify-between flex-shrink-0">
                        <div>
                            <h3 className="text-sm font-semibold text-zinc-100 font-mono">Completed Jobs</h3>
                            <div className="text-[11px] text-zinc-500 font-mono mt-0.5">{completed_count_label}</div>
                        </div>
                        <button
                            className="text-zinc-500 hover:text-zinc-200 transition-colors text-xl leading-none px-1"
                            onClick={lambda -> None { setModalOpen(False); }}
                        >×</button>
                    </div>
                    <div className="overflow-y-auto flex-1 custom-scrollbar">
                        {modal_list_rows if modal_list_rows.length > 0 else <div className="p-8 text-center text-xs text-zinc-500 font-mono">No completed tasks</div>}
                    </div>
                    <div className="px-5 py-3 border-t border-[var(--border)] flex-shrink-0">
                        <button
                            className="w-full text-xs py-1.5 rounded border border-zinc-700 hover:border-zinc-500 text-zinc-400 hover:text-zinc-200 font-mono transition-colors"
                            onClick={lambda -> None { setModalOpen(False); }}
                        >
                            Close
                        </button>
                    </div>
                </div>
            </div>;
    }

    # Only show operator in header when tasks are actively running
    queue_label = "Work Queue";
    if in_progress.length > 0 and currentOperator {
        queue_label = "Work Queue (" + currentOperator + ")";
    }

    show_empty = in_progress.length == 0 and completed.length == 0 and not isStarting;

    return
        <div>
            <div className="rounded-md border border-[var(--border)] bg-[var(--surface-0)] overflow-hidden">
                <div className="p-3 border-b border-[var(--border)] bg-zinc-900/40 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <div className="flex items-center gap-3 shrink-0 flex-wrap">
                        <h3 className="text-xs font-semibold uppercase tracking-[0.12em] text-zinc-200">
                            {queue_label}
                        </h3>
                        <button
                            onClick={lambda -> None { setCompletedHistory([]); onClearQueue(); }}
                            className="text-[10px] px-2 py-0.5 rounded border border-zinc-700 bg-zinc-900 hover:border-zinc-500 hover:text-zinc-200 text-zinc-500 transition-colors font-mono"
                        >
                            clear
                        </button>
                        {<button
                            onClick={lambda -> None { setModalOpen(True); }}
                            className="text-[10px] px-2 py-0.5 rounded border border-zinc-700 bg-zinc-900 hover:border-zinc-500 hover:text-zinc-200 text-zinc-400 transition-colors font-mono"
                        >
                            view completed
                        </button> if completed.length > 0 else <span/>}
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
                    {<div className="p-8 text-center text-xs text-zinc-200 font-mono">Queue empty</div> if show_empty else <span/>}
                    {<div className="p-4 flex items-center gap-3 font-mono text-xs text-zinc-400">
                        <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-orange)] animate-pulse flex-shrink-0"></span>
                        {<span className={"font-semibold mr-1 " + ("text-emerald-400" if startingTag == "[GEN]" else "text-blue-400")}>{startingTag}</span> if startingTag else <span/>}
                        {<span className="text-zinc-300 mr-1">{startingOp + ":"}</span> if startingOp else <span/>}
                        <span>Initializing job...</span>
                    </div> if isStarting and in_progress.length == 0 else <span/>}
                    {active_tasks_nodes if in_progress.length > 0 else <span/>}
                    {<div>
                        <div className="px-4 py-1.5 bg-zinc-900/50 border-t border-[var(--border)] flex items-center justify-between">
                            <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-600">Completed</span>
                            <span className="text-[10px] text-zinc-700 font-mono">{completed.length} kernel{("s" if completed.length != 1 else "")}</span>
                        </div>
                        {completed_summary_rows}
                    </div> if completed.length > 0 else <span/>}
                </div>
            </div>
            {modal_el}
        </div>;
}
