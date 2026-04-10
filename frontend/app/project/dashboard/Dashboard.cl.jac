"""Dashboard Page"""

import from react { useEffect, useLayoutEffect, useRef, useState }
import from "react-router-dom" { useNavigate, useParams }
import from recharts {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer
}
import from ....context.ConfigContext { useConfig }
import from ..operator_workbench.workbench.menus.GenerateModal { GenerateModal }
import from ..operator_workbench.workbench.menus.OptimizeModal { OptimizeModal }
import from ..shared.MetricsAdapter {
    to_num,
    to_list,
    list_len,
    op_matches,
    normalize_op_name,
    get_metrics_speed_source,
    build_dashboard_metrics_view,
    compute_speed_kpis
}
import from .DashboardUtils {
    format_op,
    format_op_short,
    format_tick_op,
    format_tick_ms,
    format_tick_ratio
}
import from .DashboardHeader { DashboardHeader }
import from .WorkQueuePanel { WorkQueuePanel }
import from .OperatorResultsTable { OperatorResultsTable }
import from .BenchmarkLogModal { BenchmarkLogModal }
import from .JobLogModal { JobLogModal }
import from .GpuMismatchBanner { GpuMismatchBanner }
import from .GpuMismatchModal { GpuMismatchModal }

def:pub Dashboard -> Any {
    navigate = useNavigate();
    params = useParams();
    name = params.projectName;

    config_ctx = useConfig();
    config = config_ctx.config;

    system_info_state = useState({});
    systemInfo = system_info_state[0];
    setSystemInfo = system_info_state[1];

    ops_state = useState([]);
    ops = ops_state[0];
    setOps = ops_state[1];

    loading_state = useState(True);
    loading = loading_state[0];
    setLoading = loading_state[1];

    status_state = useState({});
    status = status_state[0];
    setStatus = status_state[1];
    status_request_seq_ref = useRef(0);
    status_applied_seq_ref = useRef(0);
    status_poll_busy_ref = useRef(False);
    metrics_poll_busy_ref = useRef(False);
    mcts_poll_busy_ref = useRef(False);
    ops_poll_busy_ref = useRef(False);
    last_metrics_poll_ms_ref = useRef(0);
    last_mcts_poll_ms_ref = useRef(0);
    last_ops_poll_ms_ref = useRef(0);

    hardware_info_state = useState({});
    hardwareInfo = hardware_info_state[0];
    setHardwareInfo = hardware_info_state[1];

    mcts_summary_state = useState({});
    mctsSummary = mcts_summary_state[0];
    setMctsSummary = mcts_summary_state[1];

    bench_log_open_state = useState(False);
    benchLogOpen = bench_log_open_state[0];
    setBenchLogOpen = bench_log_open_state[1];

    generate_modal_open_state = useState(False);
    generateModalOpen = generate_modal_open_state[0];
    setGenerateModalOpen = generate_modal_open_state[1];
    generate_modal_session_state = useState(0);
    generateModalSession = generate_modal_session_state[0];
    setGenerateModalSession = generate_modal_session_state[1];
    generate_modal_session_ref = useRef(0);

    optimize_modal_open_state = useState(False);
    optimizeModalOpen = optimize_modal_open_state[0];
    setOptimizeModalOpen = optimize_modal_open_state[1];
    optimize_modal_session_state = useState(0);
    optimizeModalSession = optimize_modal_session_state[0];
    setOptimizeModalSession = optimize_modal_session_state[1];
    optimize_modal_session_ref = useRef(0);

    modal_op_name_state = useState("");
    modalOpName = modal_op_name_state[0];
    setModalOpName = modal_op_name_state[1];
    generate_modal_session_ref.current = generateModalSession;
    optimize_modal_session_ref.current = optimizeModalSession;
    op_display_order_ref = useRef({});
    next_op_display_order_ref = useRef(0);

    bench_log_loading_state = useState(False);
    benchLogLoading = bench_log_loading_state[0];
    setBenchLogLoading = bench_log_loading_state[1];

    bench_log_copied_state = useState(False);
    benchLogCopied = bench_log_copied_state[0];
    setBenchLogCopied = bench_log_copied_state[1];

    bench_log_state = useState("");
    benchLog = bench_log_state[0];
    setBenchLog = bench_log_state[1];

    log_tail_state = useState([]);
    logTail = log_tail_state[0];
    setLogTail = log_tail_state[1];

    active_log_key_state = useState("");
    activeLogKey = active_log_key_state[0];
    setActiveLogKey = active_log_key_state[1];

    log_modal_open_state = useState(False);
    logModalOpen = log_modal_open_state[0];
    setLogModalOpen = log_modal_open_state[1];

    # Speed Comparison Data State
    speedData_state = useState([]);
    speedData = speedData_state[0];
    setSpeedData = speedData_state[1];

    metrics_meta_state = useState({
        "default_mode": "deployment",
        "selection_policy": "safe",
        "speedup_label": "Estimated Op-Level Speedup"
    });
    metricsMeta = metrics_meta_state[0];
    setMetricsMeta = metrics_meta_state[1];

    op_time_mode_state = useState("best");
    opTimeMode = op_time_mode_state[0];
    setOpTimeMode = op_time_mode_state[1];

    comparison_view_state = useState("absolute");
    comparisonView = comparison_view_state[0];
    setComparisonView = comparison_view_state[1];

    forge_mode_state = useState("auto");
    forgeMode = forge_mode_state[0];
    setForgeMode = forge_mode_state[1];

    manual_ops_state = useState([]);
    manualOps = manual_ops_state[0];
    setManualOps = manual_ops_state[1];

    forge_busy_state = useState(False);
    forgeBusy = forge_busy_state[0];
    setForgeBusy = forge_busy_state[1];

    post_submit_state = useState(False);
    postSubmitActive = post_submit_state[0];
    setPostSubmitActive = post_submit_state[1];

    starting_hint_state = useState({"tag": "", "op": "", "workers": 1});
    startingHint = starting_hint_state[0];
    setStartingHint = starting_hint_state[1];

    optimistic_queue_jobs_state = useState([]);
    optimisticQueueJobs = optimistic_queue_jobs_state[0];
    setOptimisticQueueJobs = optimistic_queue_jobs_state[1];

    clear_signal_state = useState(0);
    clearSignal = clear_signal_state[0];
    setClearSignal = clear_signal_state[1];

    queue_cleared_state = useState(False);
    queueManuallyCleared = queue_cleared_state[0];
    setQueueManuallyCleared = queue_cleared_state[1];

    forge_error_state = useState("");
    forgeError = forge_error_state[0];
    setForgeError = forge_error_state[1];

    forged_tooltip_state = useState(None);
    forgedTooltipPos = forged_tooltip_state[0];
    setForgedTooltipPos = forged_tooltip_state[1];

    forge_notice_state = useState("");
    forgeNotice = forge_notice_state[0];
    setForgeNotice = forge_notice_state[1];

    gpu_mismatch_state = useState(False);
    gpuMismatch = gpu_mismatch_state[0];
    setGpuMismatch = gpu_mismatch_state[1];

    benchmarked_gpu_name_state = useState("");
    benchmarkedGpuName = benchmarked_gpu_name_state[0];
    setBenchmarkedGpuName = benchmarked_gpu_name_state[1];

    gpu_clone_busy_state = useState(False);
    gpuCloneBusy = gpu_clone_busy_state[0];
    setGpuCloneBusy = gpu_clone_busy_state[1];

    gpu_mismatch_dismissed_state = useState(False);
    gpuMismatchDismissed = gpu_mismatch_dismissed_state[0];
    setGpuMismatchDismissed = gpu_mismatch_dismissed_state[1];

    gpu_mismatch_modal_open_state = useState(False);
    gpuMismatchModalOpen = gpu_mismatch_modal_open_state[0];
    setGpuMismatchModalOpen = gpu_mismatch_modal_open_state[1];

    metricsView = build_dashboard_metrics_view(
        to_list(speedData), to_list(ops), opTimeMode, comparisonView, 8, 12
    );

    rawTimeListRows = to_list(
        metricsView["time_list_rows"]
        if metricsView and "time_list_rows" in metricsView
        else []
    );
    timeListRows = [];
    for row in rawTimeListRows {
        if not row {
            continue;
        }
        row_value = to_num(row["value"] if "value" in row else None);
        if row_value is None {
            row_value = 0;
        }
        timeListRows.push(
            {
                "rank": row["rank"] if "rank" in row else (timeListRows.length + 1),
                "name": format_op_short(str(row["name"]) if "name" in row else ""),
                "value": row_value,
                "count": row["count"] if "count" in row else 1,
                "bar_width": row["bar_width"] if "bar_width" in row else "0%"
            }
        );
    }

    effectiveComparisonView = (
        str(metricsView["effective_comparison_view"])
        if metricsView and "effective_comparison_view" in metricsView
        else comparisonView
    );
    rawSpeedChartData = to_list(
        metricsView["speed_chart_data"]
        if metricsView and "speed_chart_data" in metricsView
        else []
    );
    speedChartData = [];
    for row in rawSpeedChartData {
        if not row {
            continue;
        }
        row_name = format_op_short(str(row["name"]) if "name" in row else "");
        chart_row = {"name": row_name};
        if "ratio" in row {
            ratio_val = to_num(row["ratio"]);
            if ratio_val != None {
                chart_row["ratio"] = ratio_val;
            }
        }
        if "forged" in row {
            forged_val = to_num(row["forged"]);
            chart_row["forged"] = 0;
            if forged_val != None {
                chart_row["forged"] = forged_val;
            }
        }
        if "pytorch" in row {
            pytorch_val = to_num(row["pytorch"]);
            chart_row["pytorch"] = 0;
            if pytorch_val != None {
                chart_row["pytorch"] = pytorch_val;
            }
        }
        speedChartData.push(chart_row);
    }
    hasSpeedChart = list_len(speedChartData) > 0;

    def get_dashboard_cache_store -> dict {
        try {
            if not ("__kernelForgeDashboardCache" in window) or not window["__kernelForgeDashboardCache"] {
                window["__kernelForgeDashboardCache"] = {};
            }
            return window["__kernelForgeDashboardCache"];
        } except Exception {
            return {};
        }
    }

    def read_dashboard_cache(project_name: Any) -> dict {
        key = str(project_name or "").trim();
        if not key {
            return {};
        }
        try {
            store = get_dashboard_cache_store();
            if store and key in store and store[key] {
                return store[key];
            }
        } except Exception {
        # no-op
        }
        return {};
    }

    def write_dashboard_cache(project_name: Any, payload: dict) -> None {
        key = str(project_name or "").trim();
        if not key or not payload {
            return;
        }
        try {
            store = get_dashboard_cache_store();
            if store {
                store[key] = payload;
            }
        } except Exception {
        # no-op
        }
    }

    def persist_dashboard_cache(project_name: Any) -> None {
        write_dashboard_cache(
            project_name,
            {
                "operator_usage": ops,
                "speed_comparison": speedData,
                "metrics_meta": metricsMeta,
                "status": status,
                "hardware_info": hardwareInfo,
                "mcts_summary": mctsSummary
            }
        );
    }

    def normalize_job_status(raw: str) -> str {
        status_text = (raw or "").toLowerCase();
        if status_text == "success" or status_text == "completed" {
            return "completed";
        }
        if status_text == "error"
        or status_text == "failed"
        or status_text == "cancelled" {
            return "failed";
        }
        if status_text == "queued" {
            return "running";
        }
        return status_text if status_text else "idle";
    }

    def is_job_active(job_data: dict) -> bool {
        if not job_data {
            return False;
        }
        if "active" in job_data {
            return bool(job_data["active"]);
        }
        raw = (job_data["status"] or "").toLowerCase()
        if "status" in job_data and job_data["status"]
        else "";
        return raw == "running" or raw == "queued" or raw == "paused";
    }

    def next_status_request_seq -> int {
        next_seq = int(status_request_seq_ref.current) + 1;
        status_request_seq_ref.current = next_seq;
        return next_seq;
    }

    def apply_status_if_current(seq: int, payload: dict) -> bool {
        applied_seq = int(status_applied_seq_ref.current) if status_applied_seq_ref.current else 0;
        if seq < applied_seq {
            return False;
        }
        status_applied_seq_ref.current = seq;
        setStatus(payload or {});
        return True;
    }

    def apply_local_status_patch(payload: dict) -> None {
        seq = next_status_request_seq();
        status_applied_seq_ref.current = seq;
        setStatus(payload or {});
    }

    def queue_job_identity(job: any) -> str {
        job_key = (
            str(job["job_key"])
            if job and "job_key" in job and job["job_key"]
            else (
                "optimize"
                if job and "tag" in job and job["tag"] == "[OPT]"
                else "generate"
            )
        );
        tag = (
            str(job["tag"])
            if job and "tag" in job and job["tag"]
            else ("[OPT]" if job_key == "optimize" else "[GEN]")
        );
        ops_list = to_list(job["ops_list"]) if job and "ops_list" in job else [];
        op_name = str(ops_list[0]) if list_len(ops_list) > 0 else "";
        return job_key + "|" + tag + "|" + op_name;
    }

    def appendOptimisticQueueJob(job: dict) -> None {
        next_jobs = [];
        submit_id = str(job["submit_id"]) if job and "submit_id" in job and job["submit_id"] else "";
        replaced = False;
        for existing in to_list(optimisticQueueJobs) {
            existing_submit_id = (
                str(existing["submit_id"])
                if existing and "submit_id" in existing and existing["submit_id"]
                else ""
            );
            if submit_id and existing_submit_id == submit_id {
                next_jobs.push(job);
                replaced = True;
            } else {
                next_jobs.push(existing);
            }
        }
        if not replaced {
            next_jobs.push(job);
        }
        setOptimisticQueueJobs(next_jobs);
    }

    def removeOptimisticQueueJob(submit_id: str) -> None {
        if not submit_id {
            return;
        }
        next_jobs = [];
        for existing in to_list(optimisticQueueJobs) {
            existing_submit_id = (
                str(existing["submit_id"])
                if existing and "submit_id" in existing and existing["submit_id"]
                else ""
            );
            if existing_submit_id != str(submit_id) {
                next_jobs.push(existing);
            }
        }
        setOptimisticQueueJobs(next_jobs);
    }

    def clearStartingIfSubmit(submit_id: str) -> None {
        if not submit_id {
            return;
        }
        current_submit_id = (
            str(startingHint["submit_id"])
            if startingHint and "submit_id" in startingHint and startingHint["submit_id"]
            else ""
        );
        if postSubmitActive and current_submit_id == str(submit_id) {
            setPostSubmitActive(False);
            setStartingHint({"tag": "", "op": "", "workers": 1});
            window["__kf_post_submit"] = None;
        }
    }

    async def refreshStatusOnly -> None {
        try {
            if not name {
                return;
            }
            status_seq = next_status_request_seq();
            status_resp = await __jacSpawn("GetProjectStatus", "", {"projectName": name, "currentGpuInfo": config["selected_gpu_info"] if config and "selected_gpu_info" in config else None});
            status_data = status_resp.reports[0] if status_resp and status_resp.reports else {};
            status_applied = apply_status_if_current(status_seq, status_data or {});
            if status_applied and status_data and "gpu_mismatch" in status_data {
                setGpuMismatch(status_data["gpu_mismatch"]);
                setBenchmarkedGpuName(status_data["benchmarked_gpu_name"] if "benchmarked_gpu_name" in status_data else "");
            } elif status_applied {
                setGpuMismatch(False);
            }
            if name {
                persist_dashboard_cache(name);
            }
        } except Exception as e {
            console.error("Failed to fetch project status:", e);
        }
    }

    def get_profile_poll_ms -> int {
        profile_job = status["profile"] if status and "profile" in status else {};
        gen_job = status["generate"] if status and "generate" in status else {};
        opt_job = status["optimize"] if status and "optimize" in status else {};
        bench_job = status["benchmark"] if status and "benchmark" in status else {};
        if forgeBusy
        or is_job_active(profile_job)
        or is_job_active(gen_job)
        or is_job_active(opt_job)
        or is_job_active(bench_job) {
            return 1500;
        }
        return 15000;
    }

    def queue_state_has_op_lock(queue_state: any, op_name: str) -> bool {
        if not queue_state or not op_name {
            return False;
        }
        active_tasks_map = queue_state["active_tasks"] if "active_tasks" in queue_state and queue_state["active_tasks"] else {};
        for task_key in Object.keys(active_tasks_map) {
            task = active_tasks_map[task_key];
            task_step = (
                str(task["current_step"])
                if task and "current_step" in task and task["current_step"]
                else ""
            );
            if task_step == "Done" or task_step == "Failed" {
                continue;
            }
            task_op = "";
            if task and "op_name" in task and task["op_name"] {
                task_op = str(task["op_name"]);
            } elif str(task_key).startsWith("gen_") {
                task_op = str(task_key).substring(4);
            } elif str(task_key).startsWith("seq_opt_") {
                task_op = str(task_key).substring(8);
            }
            if task_op and op_matches(task_op, op_name) {
                return True;
            }
        }
        for queued_job in to_list(queue_state["job_queue"] if "job_queue" in queue_state else []) {
            for queued_op in to_list(queued_job["ops_list"] if queued_job and "ops_list" in queued_job else []) {
                if queued_op and op_matches(str(queued_op), op_name) {
                    return True;
                }
            }
        }
        return False;
    }

    def optimistic_queue_has_op_lock(op_name: str) -> bool {
        if not op_name {
            return False;
        }
        for queued_job in to_list(optimisticQueueJobs) {
            for queued_op in to_list(queued_job["ops_list"] if queued_job and "ops_list" in queued_job else []) {
                if queued_op and op_matches(str(queued_op), op_name) {
                    return True;
                }
            }
        }
        if (
            postSubmitActive
            and startingHint
            and "op" in startingHint
            and startingHint["op"]
            and op_matches(str(startingHint["op"]), op_name)
        ) {
            return True;
        }
        return False;
    }

    def get_stable_op_display_order(op_name: str) -> int {
        stable_key = str(name or "") + "|" + normalize_op_name(op_name);
        order_map = op_display_order_ref.current if op_display_order_ref.current else {};
        if not (stable_key in order_map) {
            next_order = int(next_op_display_order_ref.current) if next_op_display_order_ref.current else 0;
            order_map[stable_key] = next_order;
            op_display_order_ref.current = order_map;
            next_op_display_order_ref.current = next_order + 1;
        }
        return int(order_map[stable_key]);
    }

    async def openActionModal(e: any, opName: str, action: str) -> None {
        e.stopPropagation();
        if queue_state_has_op_lock(status["queue"] if status and "queue" in status else {}, opName) or optimistic_queue_has_op_lock(opName) {
            setForgeNotice("This operator already has queued or active work.");
            return;
        }
        live_gpu_info = config["selected_gpu_info"] if config and "selected_gpu_info" in config else None;
        live_resp = await __jacSpawn("GetProjectStatus", "", {"projectName": name, "currentGpuInfo": live_gpu_info});
        live_data = live_resp.reports[0] if live_resp and live_resp.reports else {};
        if queue_state_has_op_lock(live_data["queue"] if live_data and "queue" in live_data else {}, opName) or optimistic_queue_has_op_lock(opName) {
            if live_data and "gpu_mismatch" in live_data {
                setGpuMismatch(live_data["gpu_mismatch"]);
                setBenchmarkedGpuName(live_data["benchmarked_gpu_name"] if "benchmarked_gpu_name" in live_data else "");
            }
            setForgeNotice("This operator already has queued or active work.");
            return;
        }
        live_mismatch = live_data["gpu_mismatch"] if live_data and "gpu_mismatch" in live_data else False;
        if live_mismatch {
            setGpuMismatch(True);
            setBenchmarkedGpuName(live_data["benchmarked_gpu_name"] if "benchmarked_gpu_name" in live_data else "");
            setGpuMismatchModalOpen(True);
            return;
        }
        setModalOpName(opName);
        if action == "Optimize" {
            setOptimizeModalSession(lambda p: any -> any { return p + 1; });
            setOptimizeModalOpen(True);
        } else {
            setGenerateModalSession(lambda p: any -> any { return p + 1; });
            setGenerateModalOpen(True);
        }
    }

    def closeGenerateModal(session_id: any = None) -> None {
        if session_id is not None and session_id != "" {
            try {
                if int(session_id) != int(generate_modal_session_ref.current) {
                    return;
                }
            } except Exception {
            # no-op
            }
        }
        setGenerateModalOpen(False);
    }

    def closeOptimizeModal(session_id: any = None) -> None {
        if session_id is not None and session_id != "" {
            try {
                if int(session_id) != int(optimize_modal_session_ref.current) {
                    return;
                }
            } except Exception {
            # no-op
            }
        }
        setOptimizeModalOpen(False);
    }

    def openWorkbench(opName: str = "") -> None {
        if not name {
            return;
        }
        if opName {
            navigate(f"/project/{name}/operator-workbench/{opName}");
            return;
        }
        navigate(f"/project/{name}/operator-workbench");
    }

    async def handleModalStarted(hint: any) -> None {
        setQueueManuallyCleared(False);
        hint_tag = hint["_tag"] if hint and "_tag" in hint else "";
        hint_op = hint["_op"] if hint and "_op" in hint else "";
        hint_workers = hint["_workers"] if hint and "_workers" in hint else 1;
        hint_submit_id = hint["_submit_id"] if hint and "_submit_id" in hint else "";
        hint_queued = bool(hint["_queued"]) if hint and "_queued" in hint and hint["_queued"] else False;
        hint_optimistic = bool(hint["_optimistic"]) if hint and "_optimistic" in hint and hint["_optimistic"] else False;
        rollback_submit_id = hint["_rollback_submit_id"] if hint and "_rollback_submit_id" in hint else "";
        if rollback_submit_id {
            removeOptimisticQueueJob(str(rollback_submit_id));
            clearStartingIfSubmit(str(rollback_submit_id));
            setTimeout(lambda -> None { refreshStatusOnly();}, 150);
            return;
        }
        if hint_queued {
            queue_msg = hint["_message"]
            if hint and "_message" in hint and hint["_message"]
            else "Job queued.";
            appendOptimisticQueueJob({
                "queue_id": "local-" + (str(hint_submit_id) if hint_submit_id else str(Date.now())),
                "submit_id": str(hint_submit_id) if hint_submit_id else "",
                "job_key": "optimize" if hint_tag == "[OPT]" else "generate",
                "tag": hint_tag,
                "ops_list": [hint_op] if hint_op else []
            });
            clearStartingIfSubmit(str(hint_submit_id));
            setForgeNotice(queue_msg);
            if not hint_optimistic {
                setTimeout(lambda -> None { refreshStatusOnly();}, 300);
            }
            return;
        }
        removeOptimisticQueueJob(str(hint_submit_id));
        if hint_optimistic {
            setStartingHint({"tag": hint_tag, "op": hint_op, "workers": hint_workers, "submit_id": hint_submit_id});
            setPostSubmitActive(True);
            window["__kf_post_submit"] = {"tag": hint_tag, "op": hint_op, "workers": hint_workers, "submit_id": hint_submit_id, "expiry": Date.now() + 120000};
            setTimeout(lambda -> None { clearStartingIfSubmit(str(hint_submit_id));}, 120000);
            return;
        }
        setStartingHint({"tag": hint_tag, "op": hint_op, "workers": hint_workers, "submit_id": hint_submit_id});
        setPostSubmitActive(True);
        # Persist across navigation (Dashboard unmounts/remounts when switching tabs).
        window["__kf_post_submit"] = {"tag": hint_tag, "op": hint_op, "workers": hint_workers, "submit_id": hint_submit_id, "expiry": Date.now() + 120000};
        # Hard fallback: cancel after 120s if tasks never appear (e.g. job errored).
        # The useEffect below cancels it earlier once real tasks are visible.
        setTimeout(lambda -> None { clearStartingIfSubmit(str(hint_submit_id));}, 120000);
        # Kick off a single fast status RPC so polling doesn't have to wait
        # a full interval before showing the first real task.
        status_seq = next_status_request_seq();
        status_resp = await __jacSpawn("GetProjectStatus", "", {"projectName": name, "currentGpuInfo": config["selected_gpu_info"] if config and "selected_gpu_info" in config else None});
        status_data = status_resp.reports[0] if status_resp and status_resp.reports else {};
        if status_data {
            apply_status_if_current(status_seq, status_data);
        }
        setTimeout(lambda -> None { refreshStatusOnly();}, 300);
    }

    async def handleClearQueue -> None {
        # Suppress the queue display immediately — before the async backend call —
        # so stale poll responses can't cause a blink while we wait for the clear.
        setQueueManuallyCleared(True);
        setPostSubmitActive(False);
        window["__kf_post_submit"] = None;
        
        empty_queue = {
            "active_tasks": {},
            "pending_operators": [],
            "current_operator": "",
            "benchmark_slot": {"now": None, "pending": []}
        };
        patched = {**status};
        patched["queue"] = empty_queue;
        setStatus(patched);
        setClearSignal(lambda p: any -> any { return p + 1; });
        
        try {
            # Cancel running jobs and clear backend queue.
            await __jacSpawn("ClearProjectQueue", "", {"projectName": name});
        } except Exception as e {
            console.error("Failed to clear queue:", e);
            refreshData();
        }
    }

    def op_names_from_rows(op_rows: list) -> list {
        names = [];
        for op in to_list(op_rows) {
            if op and "name" in op and op["name"] {
                names.push(str(op["name"]));
            }
        }
        return names;
    }

    def toggleManualOp(op_name: str) -> None {
        next_ops = [];
        found = False;
        for existing in manualOps {
            if str(existing) == str(op_name) {
                found = True;
                continue;
            }
            next_ops.push(str(existing));
        }
        if not found {
            next_ops.push(str(op_name));
        }
        setManualOps(next_ops);
    }

    def selectAllManualOps -> None {
        setManualOps(op_names_from_rows(ops));
    }

    def clearManualOps -> None {
        setManualOps([]);
    }

    async def startForge -> None {
        if not name {
            return;
        }
        live_gpu_info = config["selected_gpu_info"] if config and "selected_gpu_info" in config else None;
        live_resp = await __jacSpawn("GetProjectStatus", "", {"projectName": name, "currentGpuInfo": live_gpu_info});
        live_data = live_resp.reports[0] if live_resp and live_resp.reports else {};
        live_mismatch = live_data["gpu_mismatch"] if live_data and "gpu_mismatch" in live_data else False;
        if live_mismatch {
            setGpuMismatch(True);
            setBenchmarkedGpuName(live_data["benchmarked_gpu_name"] if "benchmarked_gpu_name" in live_data else "");
            setGpuMismatchModalOpen(True);
            return;
        }
        setForgeError("");
        setForgeNotice("");
        setForgeBusy(True);
        try {
            resp = await __jacSpawn("StartProfile", "", {"projectName": name});
            data = resp.reports[0] if resp and resp.reports else {};
            if not data {
                setForgeError("Failed to queue profile job.");
            } elif "success" in data and not data["success"] {
                msg = data["error"] if "error" in data and data["error"] else "";
                setForgeError(msg if msg else "Failed to queue profile job.");
            } elif "skipped" in data and data["skipped"] {
                reason = data["reason"] if "reason" in data else "";
                if reason == "already_profiled" {
                    ops_resp = await __jacSpawn(
                        "GetProjectOps", "", {"projectName": name}
                    );
                    ops_data = ops_resp.reports[0]
                    if ops_resp and ops_resp.reports
                    else {};
                    ops_list = to_list(
                        ops_data["ops"] if ops_data and "ops" in ops_data else []
                    );
                    op_names = op_names_from_rows(ops_list);
                    run_ops = [];
                    if forgeMode == "manual" {
                        selected = [];
                        selected_map = {};
                        for op_name in manualOps {
                            selected_map[str(op_name)] = True;
                        }
                        for op_name in op_names {
                            if op_name in selected_map {
                                selected.push(op_name);
                            }
                        }
                        run_ops = selected;
                    } else {
                        run_ops = op_names;
                    }

                    if list_len(run_ops) == 0 {
                        if forgeMode == "manual" {
                            setForgeError("Manual mode requires at least one operator selection.");
                        } else {
                            setForgeNotice(
                                "Profile ready, but no operators were found to forge."
                            );
                        }
                    } else {
                        preferredTarget = "cuda";
                        if (
                            hardwareInfo
                            and "preferred_target" in hardwareInfo
                            and hardwareInfo["preferred_target"]
                        ) {
                            preferredTarget = hardwareInfo["preferred_target"];
                        }
                        gen_payload = {
                            "projectName": name,
                            "ops": run_ops,
                            "benchmark": True,
                            "iterations": 5,
                            "targetDevice": preferredTarget
                        };
                        gen_resp = await __jacSpawn(
                            "StartGenerate", "", gen_payload
                        );
                        gen_data = gen_resp.reports[0]
                        if gen_resp and gen_resp.reports
                        else {};
                        if gen_data
                        and "success" in gen_data
                        and gen_data["success"] {
                            setForgeNotice(
                                f"Kernel forge queued for {list_len(run_ops)} operator(s): generate + benchmark."
                            );
                            openWorkbench();
                        } else {
                            err = gen_data["error"]
                            if gen_data
                            and "error" in gen_data
                            and gen_data["error"]
                            else "Failed to queue kernel generation.";
                            setForgeError(err);
                        }
                    }
                } elif reason == "in_progress" {
                    setForgeNotice("Forge is already running.");
                } else {
                    msg = data["message"]
                    if "message" in data and data["message"]
                    else "Forge action skipped.";
                    setForgeNotice(msg);
                }
                await refreshData();
            } else {
                setForgeNotice(
                    "Forge queued: profiling and baseline benchmark started."
                );
                await refreshData();
                setTimeout(lambda -> None { refreshData();}, 1200);
                setTimeout(lambda -> None { refreshData();}, 3200);
            }
        } except Exception as e {
            setForgeError(str(e));
            console.error("Failed to start forging pipeline:", e);
        } finally {
            setForgeBusy(False);
        }
    }

    async def refreshData -> None {
        # Only show the loading skeleton when nothing is rendered yet.
        # ops/speedData are from the current render's closure -- by the time
        # useEffect fires, useLayoutEffect has already run and populated them
        # from cache (or reset them to []), so this check is always fresh.
        has_cached_data = list_len(ops) > 0 or list_len(speedData) > 0;
        if not has_cached_data {
            setLoading(True);
        }
        try {
            if name {
                status_seq = next_status_request_seq();
                # Fire all five requests in parallel so total wait is max(all five)
                # instead of max(three) + metrics + mcts sequentially.
                status_req = __jacSpawn("GetProjectStatus", "", {"projectName": name, "currentGpuInfo": config["selected_gpu_info"] if config and "selected_gpu_info" in config else None});
                ops_req = __jacSpawn("GetProjectOps", "", {"projectName": name});
                hw_req = __jacSpawn("DetectHardware", "", {});
                metrics_req = __jacSpawn("GetProjectMetrics", "", {"projectName": name});
                mcts_req = __jacSpawn("GetProjectMctsSummary", "", {"projectName": name});
                status_resp = await status_req;
                ops_resp = await ops_req;
                hw_resp = await hw_req;
                metrics_resp = await metrics_req;
                mcts_resp = await mcts_req;
                # Process Status first so queue updates are not blocked by metrics.
                status_data = status_resp.reports[0]
                if status_resp and status_resp.reports
                else {};
                status_applied = apply_status_if_current(status_seq, status_data or {});
                if status_applied and status_data and "gpu_mismatch" in status_data {
                    setGpuMismatch(status_data["gpu_mismatch"]);
                    setBenchmarkedGpuName(status_data["benchmarked_gpu_name"] if "benchmarked_gpu_name" in status_data else "");
                } elif status_applied {
                    setGpuMismatch(False);
                }
                # Process Ops
                ops_data = ops_resp.reports[0]
                if ops_resp and ops_resp.reports
                else None;
                ops_list = to_list(ops_data["ops"])
                if ops_data and "ops" in ops_data
                else [];
                setOps(ops_list);
                # Process Metrics (same source used by Operator Workbench)
                metrics_data = metrics_resp.reports[0]
                if metrics_resp and metrics_resp.reports
                else {};
                if metrics_data {
                    if "operator_usage" in metrics_data {
                        usage_list = to_list(metrics_data["operator_usage"]);
                        if usage_list.length > 0 {
                            setOps(usage_list);
                        }
                    }
                    setSpeedData(get_metrics_speed_source(metrics_data));
                    setMetricsMeta(
                        {
                            "default_mode": (
                                metrics_data["default_mode"]
                                if "default_mode" in metrics_data and metrics_data["default_mode"]
                                else "deployment"
                            ),
                            "selection_policy": (
                                metrics_data["selection_policy"]
                                if "selection_policy" in metrics_data and metrics_data["selection_policy"]
                                else "safe"
                            ),
                            "speedup_label": (
                                metrics_data["speedup_label"]
                                if "speedup_label" in metrics_data and metrics_data["speedup_label"]
                                else "Estimated Op-Level Speedup"
                            )
                        }
                    );
                }
                # Process Hardware
                hw_data = hw_resp.reports[0] if hw_resp and hw_resp.reports else {};
                setHardwareInfo(hw_data or {});
                # Process MCTS
                mcts_data = mcts_resp.reports[0]
                if mcts_resp and mcts_resp.reports
                else {};
                setMctsSummary(mcts_data or {});
                persist_dashboard_cache(name);
            }
        } except Exception as e {
            console.error("Failed to fetch dashboard data:", e);
        } finally {
            setLoading(False);
        }
    }

    useLayoutEffect(
        lambda -> None { if name {
            # Always reset data for the incoming project so that refreshData's
            # has_cached_data check never sees stale data from a previous project.
            # React batches these with the cache writes below, so no flash occurs.
            setOps([]);
            setSpeedData([]);
            cached = read_dashboard_cache(name);
            if cached {
                if "operator_usage" in cached {
                    setOps(to_list(cached["operator_usage"]));
                }
                if "speed_comparison" in cached {
                    setSpeedData(to_list(cached["speed_comparison"]));
                }
                if "metrics_meta" in cached and cached["metrics_meta"] {
                    setMetricsMeta(cached["metrics_meta"]);
                }
                if "status" in cached and cached["status"] {
                    setStatus(cached["status"]);
                }
                if "hardware_info" in cached and cached["hardware_info"] {
                    setHardwareInfo(cached["hardware_info"]);
                }
                if "mcts_summary" in cached and cached["mcts_summary"] {
                    setMctsSummary(cached["mcts_summary"]);
                }
                setLoading(False);
            }
        }},
        [name]
    );

    useEffect(lambda -> None { refreshData();}, [name]);

    # Stable boolean dep for the polling effect — only flips when jobs start/stop,
    # not on every status object update.  Computed before the effect so it is
    # defined once per render and never reassigned (avoids var-scoping issues).
    _ep_profile = status["profile"] if status and "profile" in status else {};
    _ep_gen = status["generate"] if status and "generate" in status else {};
    _ep_opt = status["optimize"] if status and "optimize" in status else {};
    _ep_bench = status["benchmark"] if status and "benchmark" in status else {};
    _ep_queue_data = status["queue"] if status and "queue" in status else {};
    _ep_active_tasks = _ep_queue_data["active_tasks"] if "active_tasks" in _ep_queue_data else {};
    _ep_job_queue = _ep_queue_data["job_queue"] if "job_queue" in _ep_queue_data else [];
    merged_job_queue = [];
    for real_job in to_list(_ep_job_queue) {
        merged_job_queue.push(real_job);
    }
    for local_job in to_list(optimisticQueueJobs) {
        local_identity = queue_job_identity(local_job);
        found_match = False;
        for real_job in to_list(_ep_job_queue) {
            if queue_job_identity(real_job) == local_identity {
                found_match = True;
                break;
            }
        }
        if not found_match {
            merged_job_queue.push(local_job);
        }
    }
    has_active_queue_tasks = False;
    for active_task_key in Object.keys(_ep_active_tasks) {
        active_task = _ep_active_tasks[active_task_key];
        active_task_step = (
            str(active_task["current_step"])
            if active_task and "current_step" in active_task and active_task["current_step"]
            else ""
        );
        if active_task_step != "Done" and active_task_step != "Failed" {
            has_active_queue_tasks = True;
            break;
        }
    }
    has_queue_work = has_active_queue_tasks or list_len(merged_job_queue) > 0;
    any_job_active = postSubmitActive or forgeBusy or has_queue_work or is_job_active(_ep_profile) or is_job_active(_ep_gen) or is_job_active(_ep_opt) or is_job_active(_ep_bench);

    # Capture current GPU info for polling closure (must be in deps so closure stays fresh)
    _poll_gpu_info = config["selected_gpu_info"] if config and "selected_gpu_info" in config else None;

    # Auto-refresh status (faster while active, slower while idle)
    useEffect(
        lambda -> Any { poll_ms = 500 if any_job_active else 15000;timer = setInterval(
            lambda -> None {
                if not name or status_poll_busy_ref.current {
                    return;
                }
                status_poll_busy_ref.current = True;
                status_seq = next_status_request_seq();
                __jacSpawn("GetProjectStatus", "", {"projectName": name, "currentGpuInfo": _poll_gpu_info}).then(
                lambda resp: any  -> None { data = resp.reports[0]
                if resp and resp.reports
                else {};
                status_applied = apply_status_if_current(status_seq, data or {});
                if status_applied and data and "gpu_mismatch" in data {
                    setGpuMismatch(data["gpu_mismatch"]);
                    setBenchmarkedGpuName(data["benchmarked_gpu_name"] if "benchmarked_gpu_name" in data else "");
                } elif status_applied {
                    setGpuMismatch(False);
                }
                now_ms = Date.now();
                if not any_job_active
                and not metrics_poll_busy_ref.current
                and now_ms - int(last_metrics_poll_ms_ref.current) >= 5000 {
                    metrics_poll_busy_ref.current = True;
                    last_metrics_poll_ms_ref.current = now_ms;
                    __jacSpawn("GetProjectMetrics", "", {"projectName": name}).then(
                    lambda c_resp: any  -> None { c_data = c_resp.reports[0]
                    if c_resp and c_resp.reports
                    else {};
                    if c_data {
                        if "operator_usage" in c_data {
                            setOps(to_list(c_data["operator_usage"]));
                        }
                        setSpeedData(get_metrics_speed_source(c_data));
                        persist_dashboard_cache(name);
                    }
                    metrics_poll_busy_ref.current = False;}
                ).catch(
                    lambda e: any  -> None {
                    metrics_poll_busy_ref.current = False;
                    # console.error("Failed to poll project metrics:", e);
                    }
                );
                }
                if not any_job_active
                and not mcts_poll_busy_ref.current
                and now_ms - int(last_mcts_poll_ms_ref.current) >= 5000 {
                    mcts_poll_busy_ref.current = True;
                    last_mcts_poll_ms_ref.current = now_ms;
                    __jacSpawn("GetProjectMctsSummary", "", {"projectName": name}).then(
                    lambda m_resp: any  -> None { m_data = m_resp.reports[0]
                    if m_resp and m_resp.reports
                    else {};
                    setMctsSummary(m_data or {});
                    persist_dashboard_cache(name);
                    mcts_poll_busy_ref.current = False;}
                ).catch(
                    lambda e: any  -> None {
                    mcts_poll_busy_ref.current = False;
                    # console.error("Failed to poll mcts summary:", e);
                    }
                );
                }
                profile_is_active = data and "profile" in data and data["profile"] and "active" in data["profile"] and data["profile"]["active"];
                if profile_is_active
                and not ops_poll_busy_ref.current
                and now_ms - int(last_ops_poll_ms_ref.current) >= 5000 {
                    ops_poll_busy_ref.current = True;
                    last_ops_poll_ms_ref.current = now_ms;
                    __jacSpawn("GetProjectOps", "", {"projectName": name}).then(
                    lambda o_resp: any -> None { o_data = o_resp.reports[0]
                    if o_resp and o_resp.reports
                    else {};
                    ops_list = to_list(o_data["ops"] if o_data and "ops" in o_data else []);
                    if ops_list.length > 0 {
                        setOps(ops_list);
                    }
                    ops_poll_busy_ref.current = False;}
                ).catch(
                    lambda e: any -> None {
                    ops_poll_busy_ref.current = False;
                    }
                );
                }
                log_job_key = "";
                if data and "generate" in data and data["generate"] and "active" in data["generate"] and data["generate"]["active"] {
                    log_job_key = "generate";
                } elif data and "optimize" in data and data["optimize"] and "active" in data["optimize"] and data["optimize"]["active"] {
                    log_job_key = "optimize";
                } elif data and "profile" in data and data["profile"] and "active" in data["profile"] and data["profile"]["active"] {
                    log_job_key = "profile";
                } elif data and "benchmark" in data and data["benchmark"] and "active" in data["benchmark"] and data["benchmark"]["active"] {
                    log_job_key = "benchmark";
                }
                if log_job_key {
                    setActiveLogKey(log_job_key);
                    __jacSpawn("GetJobLog", "", {"projectName": name, "jobKey": log_job_key, "lines": 12}).then(
                        lambda l_resp: any -> None { l_data = l_resp.reports[0]
                        if l_resp and l_resp.reports
                        else {};
                        setLogTail(l_data["lines"] if l_data and "lines" in l_data else []);}
                    ).catch(lambda e: any -> None {});
                } else {
                    setLogTail([]);
                    setActiveLogKey("");
                }
                status_poll_busy_ref.current = False;}
            ).catch(
                lambda e: any  -> None {
                status_poll_busy_ref.current = False;
                # console.error("Failed to poll project status:", e);
                }
            );},
            poll_ms
        );return lambda -> None { clearInterval(timer);}; },
        [name, forgeBusy, any_job_active, _poll_gpu_info]
    );

    # On page refresh, config loads after the first poll fires (with null GPU info).
    # Fire one immediate status call whenever _poll_gpu_info becomes available.
    useEffect(
        lambda -> None {
            if _poll_gpu_info and name {
                __jacSpawn("GetProjectStatus", "", {"projectName": name, "currentGpuInfo": _poll_gpu_info}).then(
                    lambda resp: any -> None {
                        data = resp.reports[0] if resp and resp.reports else {};
                        if data and "gpu_mismatch" in data {
                            setGpuMismatch(data["gpu_mismatch"]);
                            setBenchmarkedGpuName(data["benchmarked_gpu_name"] if "benchmarked_gpu_name" in data else "");
                        } else {
                            setGpuMismatch(False);
                        }
                    }
                ).catch(lambda e: any -> None {});
            }
        },
        [_poll_gpu_info, name]
    );

    # Restore "starting" indicator if navigated back before tasks appeared.
    useEffect(
        lambda -> None {
            try {
                stored = window["__kf_post_submit"] if "__kf_post_submit" in window else None;
                if stored and stored["expiry"] > Date.now() {
                    setStartingHint({
                        "tag": stored["tag"],
                        "op": stored["op"],
                        "workers": stored["workers"] if "workers" in stored else 1,
                        "submit_id": stored["submit_id"] if "submit_id" in stored else ""
                    });
                    setPostSubmitActive(True);
                }
            } except Exception {}
        },
        [name]
    );

    # Cancel the "starting" indicator the moment real queue tasks appear.
    useEffect(
        lambda -> None { if postSubmitActive and has_active_queue_tasks { setPostSubmitActive(False); setStartingHint({"tag": "", "op": "", "workers": 1, "submit_id": ""}); window["__kf_post_submit"] = None; } },
        [postSubmitActive, has_active_queue_tasks]
    );

    useEffect(
        lambda -> None {
            next_jobs = [];
            for local_job in to_list(optimisticQueueJobs) {
                local_identity = queue_job_identity(local_job);
                matched = False;
                for real_job in to_list(_ep_job_queue) {
                    if queue_job_identity(real_job) == local_identity {
                        matched = True;
                        break;
                    }
                }
                if not matched {
                    for task in to_list(_ep_active_tasks) {
                        task_id = str(task["id"]) if task and "id" in task and task["id"] else "";
                        task_tag = (
                            str(task["tag"])
                            if task and "tag" in task and task["tag"]
                            else ("[GEN]" if task_id.startsWith("gen_") else "[OPT]")
                        );
                        task_job_key = "optimize" if task_tag == "[OPT]" else "generate";
                        task_op = str(task["op_name"]) if task and "op_name" in task and task["op_name"] else "";
                        if local_identity == task_job_key + "|" + task_tag + "|" + task_op {
                            matched = True;
                            break;
                        }
                    }
                }
                if not matched {
                    next_jobs.push(local_job);
                }
            }
            if list_len(next_jobs) != list_len(optimisticQueueJobs) {
                setOptimisticQueueJobs(next_jobs);
            }
        },
        [_ep_job_queue, _ep_active_tasks, optimisticQueueJobs]
    );

    async def fetchSystemInfo -> None {
        try {
            active_idx = config["active_ssh_index"]
            if config and "active_ssh_index" in config
            else -1;
            ssh_connections = config["ssh_connections"]
            if config and "ssh_connections" in config
            else [];
            ssh_conn = None;
            if active_idx != -1 and active_idx < list_len(ssh_connections) {
                ssh_conn = ssh_connections[active_idx];
            }

            payload = {};
            if ssh_conn {
                payload = {
                    "host": ssh_conn["host"] if "host" in ssh_conn else "",
                    "user": ssh_conn["user"] if "user" in ssh_conn else "",
                    "password": "",
                    "key_path": ssh_conn["key_path"] if "key_path" in ssh_conn else ""
                };
            }

            resp = await __jacSpawn("GetSystemInfo", "", payload);
            data = resp.reports[0] if resp and resp.reports else {};

            if data {
                setSystemInfo(data);
            }
        } except Exception as e {
            console.error("Failed to fetch system info:", e);
        }
    }

    useEffect(lambda -> None { fetchSystemInfo();}, [config]);

    useEffect(
        lambda -> None {
            if gpuMismatch and benchmarkedGpuName and name {
                key = f"kf_gpu_dismiss_{name}_{benchmarkedGpuName}";
                if localStorage.getItem(key) {
                    setGpuMismatchDismissed(True);
                } else {
                    setGpuMismatchDismissed(False);
                }
            } else {
                setGpuMismatchDismissed(False);
            }
        },
        [gpuMismatch, benchmarkedGpuName, name]
    );

    async def handleCloneForGpu -> None {
        setGpuCloneBusy(True);
        try {
            resp = await __jacSpawn("CloneProjectForGpu", "", {"sourceProjectName": name, "currentGpuInfo": config["selected_gpu_info"] if config and "selected_gpu_info" in config else None});
            data = resp.reports[0] if resp and resp.reports else {};
            if data and "success" in data and data["success"] {
                navigate(f"/project/{data['newProjectName']}");
            } else {
                setForgeError("Clone failed.");
            }
        } except Exception as e {
            setForgeError("Clone failed.");
        } finally {
            setGpuCloneBusy(False);
        }
    }

    def handleDismissGpuMismatch -> None {
        setGpuMismatchDismissed(True);
        if name and benchmarkedGpuName {
            key = f"kf_gpu_dismiss_{name}_{benchmarkedGpuName}";
            localStorage.setItem(key, "1");
        }
    }

    useEffect(
        lambda -> None { if benchLogOpen and name {
            refreshBenchLog();
        }},
        [benchLogOpen, name]
    );

    async def refreshBenchLog -> None {
        if not name {
            return;
        }
        setBenchLogLoading(True);
        setBenchLogCopied(False);
        try {
            rel_path = f"kernels/projects/{name}/logs/benchmark.log";
            resp = await __jacSpawn(
                "ReadRepoFile", "", {"projectName": name, "relPath": rel_path}
            );
            data = resp.reports[0] if resp and resp.reports else {};
            if data and "success" in data and data["success"] {
                setBenchLog(data["content"] if "content" in data else "");
            } else {
                setBenchLog(data["error"] if data and "error" in data else "No log yet.");
            }
        } except Exception as e {
            setBenchLog(f"Failed to load benchmark log: {e}");
        } finally {
            setBenchLogLoading(False);
        }
    }

    def copyBenchLog -> None {
        if not benchLog {
            return;
        }
        navigator.clipboard.writeText(benchLog).then(
            lambda -> None { setBenchLogCopied(True);setTimeout(
                lambda -> None { setBenchLogCopied(False);}, 1500
            );}
        ).catch(
            lambda e: any  -> None { console.error("Failed to copy benchmark log:", e);}
        );
    }

    # Helper for hardware status
    hardwareStatus = "Scanning...";
    activeBackend = "CPU";
    gpu_name = "";
    gpus = to_list(systemInfo["gpus"] if systemInfo and "gpus" in systemInfo else []);
    selected_idx = config["selected_gpu_index"]
    if config and "selected_gpu_index" in config
    else -1;

    # Detailed GPU stats
    gpu_mem = "";
    gpu_compute = "";
    gpu_sms = "";

    display_gpu = None;
    auto_gpu = False;
    if config and "selected_gpu_info" in config and config["selected_gpu_info"] {
        display_gpu = config["selected_gpu_info"];
    } elif selected_idx != -1 and selected_idx < list_len(gpus) {
        display_gpu = gpus[selected_idx];
    } elif list_len(gpus) > 0 {
        display_gpu = gpus[0];
        auto_gpu = True;
    }

    if display_gpu {
        conn = display_gpu["connection"] if "connection" in display_gpu else "local";
        gpu_name = display_gpu["name"] if "name" in display_gpu else "Unknown GPU";
        gpu_mem = display_gpu["memory"] if "memory" in display_gpu else "";
        gpu_compute = display_gpu["compute_capability"]
        if "compute_capability" in display_gpu
        else "";
        gpu_sms = f"{display_gpu['sm_count']}" if "sm_count" in display_gpu else "";
        suffix = " (Auto)" if auto_gpu else "";
        hardwareStatus = f"{gpu_name}{suffix}";
        if conn and conn != "local" {
            hardwareStatus = f"{hardwareStatus} ({conn})";
        }
        activeBackend = "CUDA";
    } else {
        # Fallback to detection if we have no detailed GPU list
        if hardwareInfo
        and "mps_available" in hardwareInfo
        and hardwareInfo["mps_available"] {
            hardwareStatus = "MPS: Apple Silicon";
            activeBackend = "MPS";
        } elif hardwareInfo and "cuda_available" in hardwareInfo {
            if hardwareInfo["cuda_available"] {
                gpu_name = hardwareInfo["device_name"] or "GPU";
                hardwareStatus = f"CUDA: {gpu_name} (Auto)";
                activeBackend = "CUDA";
            } else {
                hardwareStatus = "CPU Only";
                activeBackend = "CPU";
            }
        } else {
            hardwareStatus = "Scanning...";
            activeBackend = "CPU";
        }
    }

    # Pipeline job state
    profile_job = status["profile"] if status and "profile" in status else {};
    profile_raw = (profile_job["status"] or "").toLowerCase()
    if profile_job and "status" in profile_job
    else "";
    profile_phase = (profile_job["phase"] or "").toLowerCase()
    if profile_job and "phase" in profile_job
    else "";
    profile_message = profile_job["message"]
    if profile_job and "message" in profile_job and profile_job["message"]
    else "";
    profile_active = is_job_active(profile_job);
    profile_progress_pct = parseInt(f"{profile_job['progress_percent'] * 100}")
    if profile_job and "progress_percent" in profile_job
    else 0;
    if profile_progress_pct < 0 {
        profile_progress_pct = 0;
    }
    if profile_progress_pct > 100 {
        profile_progress_pct = 100;
    }
    profile_norm = normalize_job_status(profile_raw);

    bench_job = status["benchmark"] if status and "benchmark" in status else {};
    bench_raw = (bench_job["status"] or "").toLowerCase()
    if bench_job and "status" in bench_job
    else "";
    bench_message = bench_job["message"]
    if bench_job and "message" in bench_job and bench_job["message"]
    else "";
    bench_norm = normalize_job_status(bench_raw);
    bench_active = is_job_active(bench_job);

    bench_progress = bench_job["progress"]
    if bench_job and "progress" in bench_job
    else {};
    bench_current = parseInt(f"{bench_progress['current']}")
    if bench_progress and "current" in bench_progress
    else 0;
    bench_total = parseInt(f"{bench_progress['total']}")
    if bench_progress and "total" in bench_progress
    else 0;
    bench_progress_text = (
        f" ({bench_current}/{bench_total})" if bench_total and bench_total > 0 else ""
    );
    gen_job = status["generate"] if status and "generate" in status else {};

    gen_progress = gen_job["progress"] if gen_job and "progress" in gen_job else {};
    gen_current = parseInt(f"{gen_progress['current']}")
    if gen_progress and "current" in gen_progress
    else 0;
    gen_total = parseInt(f"{gen_progress['total']}")
    if gen_progress and "total" in gen_progress
    else 0;
    gen_raw = (gen_job["status"] or "").toLowerCase()
    if gen_job and "status" in gen_job
    else "";
    gen_active = is_job_active(gen_job);
    gen_message = gen_job["message"]
    if gen_job and "message" in gen_job and gen_job["message"]
    else "";
    gen_phase = (gen_job["phase"] or "").toLowerCase()
    if gen_job and "phase" in gen_job
    else "";
    gen_has_attempt_msg = gen_message and gen_message.includes("attempt");
    gen_progress_text = (
        f" — {gen_current}/{gen_total} done"
        if (gen_total and gen_total > 1 and not gen_has_attempt_msg)
        else ""
    );
    gen_norm = normalize_job_status(gen_raw);
    gen_cmd = to_list(gen_job["command"] if gen_job and "command" in gen_job else []);
    gen_has_optimize = False;
    gen_has_benchmark = False;
    for part in gen_cmd {
        token = str(part);
        if token == "--optimize" {
            gen_has_optimize = True;
        } elif token == "--benchmark" {
            gen_has_benchmark = True;
        }
    }

    opt_job = status["optimize"] if status and "optimize" in status else {};
    opt_raw = (opt_job["status"] or "").toLowerCase()
    if opt_job and "status" in opt_job
    else "";
    opt_active = is_job_active(opt_job);
    opt_message = opt_job["message"]
    if opt_job and "message" in opt_job and opt_job["message"]
    else "";
    opt_norm = normalize_job_status(opt_raw);
    opt_progress = opt_job["progress"] if opt_job and "progress" in opt_job else {};
    opt_current = parseInt(f"{opt_progress['current']}")
    if opt_progress and "current" in opt_progress
    else 0;
    opt_total = parseInt(f"{opt_progress['total']}")
    if opt_progress and "total" in opt_progress
    else 0;
    opt_progress_text = (
        f" — {opt_current}/{opt_total} done" if opt_total and opt_total > 1 else ""
    );

    active_poll = forgeBusy
    or profile_active
    or gen_active
    or opt_active
    or bench_active;
    poll_label = "1.5s (active)" if active_poll else "15s (idle)";

    generation_step_status = gen_norm;
    if generation_step_status == "running" and gen_phase == "benchmarking" {
        generation_step_status = "completed";
    }

    optimization_step_status = opt_norm;
    if optimization_step_status == "idle" {
        if gen_phase == "optimizing" and gen_norm == "running" {
            optimization_step_status = "running";
        } elif gen_has_optimize
        and (
            (gen_phase == "benchmarking" and gen_norm == "running")
            or gen_norm == "completed"
        ) {
            optimization_step_status = "completed";
        } elif gen_has_optimize and gen_norm == "failed" {
            optimization_step_status = "failed";
        }
    }

    benchmark_step_status = bench_norm;
    if benchmark_step_status == "idle" and gen_has_benchmark {
        if gen_phase == "benchmarking" and gen_norm == "running" {
            benchmark_step_status = "running";
        } elif gen_norm == "completed" {
            benchmark_step_status = "completed";
        } elif gen_norm == "failed" and gen_phase == "benchmarking" {
            benchmark_step_status = "failed";
        }
    }

    current_phase = "idle";
    current_message = "Waiting to start forge.";
    current_progress_pct = 0;
    current_started_at = "";
    current_updated_at = "";

    if loading {
        current_phase = "loading";
        current_message = "Syncing forge state from backend...";
    } elif forgeBusy {
        current_phase = "queued";
        current_message = "Queuing forge pipeline.";
    } elif gen_active {
        current_phase = gen_phase if gen_phase else "generating";
        current_message = (gen_message + gen_progress_text) if gen_message else ("Generating kernels." + gen_progress_text);
        current_progress_pct = parseInt(f"{gen_job['progress_percent'] * 100}")
        if gen_job and "progress_percent" in gen_job
        else 0;
        current_started_at = gen_job["started_at"]
        if gen_job and "started_at" in gen_job and gen_job["started_at"]
        else "";
        current_updated_at = gen_job["updated_at"]
        if gen_job and "updated_at" in gen_job and gen_job["updated_at"]
        else "";
    } elif opt_active {
        current_phase = "optimizing";
        current_message = (opt_message + opt_progress_text) if opt_message else ("Optimizing kernels." + opt_progress_text);
        current_progress_pct = parseInt(f"{opt_job['progress_percent'] * 100}")
        if opt_job and "progress_percent" in opt_job
        else 0;
        current_started_at = opt_job["started_at"]
        if opt_job and "started_at" in opt_job and opt_job["started_at"]
        else "";
        current_updated_at = opt_job["updated_at"]
        if opt_job and "updated_at" in opt_job and opt_job["updated_at"]
        else "";
    } elif bench_active {
        current_phase = "benchmarking";
        current_message = bench_message if bench_message else "Benchmarking kernels.";
        current_progress_pct = parseInt(f"{bench_job['progress_percent'] * 100}")
        if bench_job and "progress_percent" in bench_job
        else 0;
        current_started_at = bench_job["started_at"]
        if bench_job and "started_at" in bench_job and bench_job["started_at"]
        else "";
        current_updated_at = bench_job["updated_at"]
        if bench_job and "updated_at" in bench_job and bench_job["updated_at"]
        else "";
    } elif profile_active {
        current_phase = profile_phase if profile_phase else "profiling";
        current_message = (
            profile_message
            if profile_message
            else "Profiling and baseline benchmarking in progress."
        );
        current_progress_pct = profile_progress_pct;
        current_started_at = profile_job["started_at"]
        if profile_job and "started_at" in profile_job and profile_job["started_at"]
        else "";
        current_updated_at = profile_job["updated_at"]
        if profile_job and "updated_at" in profile_job and profile_job["updated_at"]
        else "";
    } elif benchmark_step_status == "completed" {
        gen_had_failures = (
            bench_norm == "idle"
            and gen_norm == "completed"
            and gen_has_benchmark
            and gen_message
            and gen_message.includes("Unable to forge")
        );
        if gen_had_failures {
            current_phase = "completed_with_failures";
            current_message = gen_message;
        } else {
            current_phase = "completed";
            current_message = gen_message if gen_message else "Kernel generation and benchmark completed.";
        }
    } elif profile_norm == "failed"
    or generation_step_status == "failed"
    or optimization_step_status == "failed"
    or benchmark_step_status == "failed" {
        current_phase = "failed";
        failure_detail = "";
        if generation_step_status == "failed" and gen_message {
            failure_detail = gen_message;
        } elif optimization_step_status == "failed" and opt_message {
            failure_detail = opt_message;
        } elif benchmark_step_status == "failed" and bench_message {
            failure_detail = bench_message;
        } elif profile_norm == "failed" and profile_message {
            failure_detail = profile_message;
        }
        current_message = (
            failure_detail if failure_detail else "Forge run failed. See logs for details."
        );
    } elif profile_norm == "completed" {
        current_phase = "ready";
        current_message = "Profile ready. Start kernel generation from dashboard or workbench.";
    }

    if current_progress_pct < 0 {
        current_progress_pct = 0;
    }
    if current_progress_pct > 100 {
        current_progress_pct = 100;
    }
    current_progress_bar = current_progress_pct;
    if active_poll and current_progress_bar < 4 {
        current_progress_bar = 4;
    }

    forge_status = "Idle";
    if current_phase == "queued" {
        forge_status = "Heating up the forge";
    } elif current_phase == "loading" {
        forge_status = "Syncing state";
    } elif current_phase == "preparing" {
        forge_status = "Preparing assets";
    } elif current_phase == "profiling" {
        forge_status = "Profiling operators";
    } elif current_phase == "generating" {
        forge_status = "Generating kernels";
    } elif current_phase == "optimizing" {
        forge_status = "Optimizing kernels";
    } elif current_phase == "benchmarking" {
        forge_status = "Benchmarking kernels";
    } elif current_phase == "completed" {
        forge_status = "Forge completed";
    } elif current_phase == "completed_with_failures" {
        forge_status = "Forge completed (partial)";
    } elif current_phase == "failed" {
        forge_status = "Forge run failed";
    } elif current_phase == "ready" {
        forge_status = "Ready for operator generation";
    }

    forge_detail = current_message if current_message else "";
    if not forge_detail and forgeNotice {
        forge_detail = forgeNotice;
    }
    if not forge_detail {
        forge_detail = "Idle";
    }

    def derive_task_tag(task: any) -> str {
        task_id = str(task["id"]) if task and "id" in task and task["id"] else "";
        if task and "tag" in task and task["tag"] {
            tag_text = str(task["tag"]).trim();
            if tag_text and tag_text != "undefined" and tag_text != "null" {
                return tag_text;
            }
        }
        return "[GEN]" if task_id.startsWith("gen_") else "[OPT]";
    }

    def derive_task_op_name(task: any) -> str {
        if task and "op_name" in task and task["op_name"] {
            op_text = str(task["op_name"]).trim();
            if op_text and op_text != "undefined" and op_text != "null" {
                return op_text;
            }
        }
        task_id = str(task["id"]) if task and "id" in task and task["id"] else "";
        if task_id.startsWith("gen_") {
            return task_id.substring(4);
        }
        if task_id.startsWith("seq_opt_") {
            return task_id.substring(8);
        }
        return "";
    }

    def find_queue_summary_for_op(queue_summary: dict, op_name: str) -> dict {
        direct_key = normalize_op_name(op_name);
        if direct_key and direct_key in queue_summary and queue_summary[direct_key] {
            return queue_summary[direct_key];
        }
        for summary_key in Object.keys(queue_summary) {
            if op_matches(summary_key, op_name) {
                return queue_summary[summary_key];
            }
        }
        return {};
    }

    def queued_job_contains_op(job: any, op_name: str) -> bool {
        if not job or not op_name {
            return False;
        }
        for queued_op in to_list(job["ops_list"] if "ops_list" in job else []) {
            if queued_op and op_matches(str(queued_op), op_name) {
                return True;
            }
        }
        return False;
    }

    queue_data = status["queue"] if status and "queue" in status else {};
    active_tasks_dict = queue_data["active_tasks"] if "active_tasks" in queue_data else {};
    active_tasks = [];
    for k in Object.keys(active_tasks_dict) {
        t = active_tasks_dict[k];
        t["id"] = k;
        active_tasks.push(t);
    }
    benchmark_slot = queue_data["benchmark_slot"] if "benchmark_slot" in queue_data else {"now": None, "pending": []};
    pending_operators = queue_data["pending_operators"] if "pending_operators" in queue_data else [];
    current_operator = queue_data["current_operator"] if "current_operator" in queue_data else "";
    queued_jobs = merged_job_queue;
    # While the user has manually cleared the queue, pass empty arrays so
    # WorkQueuePanel shows nothing regardless of stale poll responses.
    active_tasks_for_queue = [] if queueManuallyCleared else active_tasks;

    forgedCount = 0;
    totalPytorchMs = 0.0;
    totalForgedMs = 0.0;
    totalAllPytorchMs = 0.0;
    operatorRows = [];
    queue_summary_by_op = {};
    locked_op_names = [];

    for task in active_tasks {
        task_op_name = derive_task_op_name(task);
        task_key = normalize_op_name(task_op_name);
        if not task_key {
            continue;
        }
        if not (task_key in queue_summary_by_op) {
            queue_summary_by_op[task_key] = {
                "active_step": "",
                "has_failed": False,
                "has_gen_done": False,
                "has_opt_done": False,
                "best_gen_ms": None,
                "best_opt_ms": None
            };
        }
        entry = queue_summary_by_op[task_key];
        task_step = str(task["current_step"]) if task and "current_step" in task and task["current_step"] else "";
        task_tag = derive_task_tag(task);
        if not task_step {
            continue;
        }
        if task_step != "Done" and task_step != "Failed" {
            entry["active_step"] = task_step;
            if task_op_name {
                locked_op_names.push(task_op_name);
            }
            continue;
        }
        if task_step == "Failed" {
            entry["has_failed"] = True;
            continue;
        }
        task_value_ms = to_num(task["value_ms"] if task and "value_ms" in task else None);
        if task_tag == "[OPT]" {
            entry["has_opt_done"] = True;
            if task_value_ms != None and (entry["best_opt_ms"] == None or task_value_ms < entry["best_opt_ms"]) {
                entry["best_opt_ms"] = task_value_ms;
            }
        } else {
            entry["has_gen_done"] = True;
            if task_value_ms != None and (entry["best_gen_ms"] == None or task_value_ms < entry["best_gen_ms"]) {
                entry["best_gen_ms"] = task_value_ms;
            }
        }
    }

    for queued_job in to_list(queued_jobs) {
        for queued_op in to_list(queued_job["ops_list"] if queued_job and "ops_list" in queued_job else []) {
            if queued_op {
                locked_op_names.push(str(queued_op));
            }
        }
    }
    if postSubmitActive and startingHint and "op" in startingHint and startingHint["op"] {
        locked_op_names.push(str(startingHint["op"]));
    }

    for op in to_list(ops) {
        if not op or not ("name" in op) or not op["name"] {
            continue;
        }
        op_name = str(op["name"]);
        op_short = format_op_short(op_name);

        op_count_raw = op["count"] if "count" in op else 1;
        op_count_n = parseInt(f"{op_count_raw}") if op_count_raw else 1;
        if not op_count_n or op_count_n <= 0 { op_count_n = 1; }

        baseline_ms = "--";
        best_ms = "--";
        speedup = "--";
        is_forged = False;
        ratio_for_sort = 0.0;
        b_val_raw = None;
        metrics_best_raw = None;
        op_queue_summary = find_queue_summary_for_op(queue_summary_by_op, op_name);
        queue_gen_best_raw = (
            to_num(op_queue_summary["best_gen_ms"])
            if op_queue_summary and "best_gen_ms" in op_queue_summary
            else None
        );
        queue_opt_best_raw = (
            to_num(op_queue_summary["best_opt_ms"])
            if op_queue_summary and "best_opt_ms" in op_queue_summary
            else None
        );

        for sd in to_list(speedData) {
            if sd and "name" in sd and op_matches(str(sd["name"]), op_name) {
                b_val = to_num(sd["pytorch"] if "pytorch" in sd else None);
                b_val_raw = b_val;
                if b_val != None { baseline_ms = format_tick_ms(b_val); }

                f_val = to_num(sd["forged"] if "forged" in sd else None);
                metrics_best_raw = f_val;
                break;
            }
        }
        # Fallback: use profile_ms from GetProjectOps if speedData had no match
        if b_val_raw == None and "profile_ms" in op {
            fb_val = to_num(op["profile_ms"]);
            if fb_val != None and fb_val > 0 {
                b_val_raw = fb_val;
                baseline_ms = format_tick_ms(fb_val);
            }
        }

        if b_val_raw != None and b_val_raw > 0 {
            totalAllPytorchMs = totalAllPytorchMs + b_val_raw * op_count_n;
        }

        display_best_raw = metrics_best_raw;
        if display_best_raw == None {
            display_best_raw = queue_opt_best_raw if queue_opt_best_raw != None else queue_gen_best_raw;
        }
        if display_best_raw != None {
            best_ms = format_tick_ms(display_best_raw);
        }

        if b_val_raw != None and b_val_raw > 0 and display_best_raw != None and display_best_raw > 0 {
            display_ratio = b_val_raw / display_best_raw;
            speedup = f"{display_ratio.toFixed(2)}×";
            ratio_for_sort = display_ratio;
        }

        forged_best_raw = metrics_best_raw if metrics_best_raw != None else queue_opt_best_raw;
        if b_val_raw != None and b_val_raw > 0 and forged_best_raw != None and forged_best_raw > 0 {
            forged_ratio = b_val_raw / forged_best_raw;
            if forged_ratio > 1.001 {
                is_forged = True;
                totalPytorchMs = totalPytorchMs + b_val_raw * op_count_n;
                totalForgedMs = totalForgedMs + forged_best_raw * op_count_n;
            }
        }

        op_status = (
            str(op_queue_summary["active_step"])
            if op_queue_summary and "active_step" in op_queue_summary and op_queue_summary["active_step"]
            else ""
        );
        if not op_status {
            if is_forged {
                op_status = "Forged ✓";
            } elif (
                display_best_raw != None
                or (op_queue_summary and "has_gen_done" in op_queue_summary and op_queue_summary["has_gen_done"])
                or (op_queue_summary and "has_opt_done" in op_queue_summary and op_queue_summary["has_opt_done"])
            ) {
                op_status = "Benchmarked";
            } elif op_queue_summary and "has_failed" in op_queue_summary and op_queue_summary["has_failed"] {
                op_status = "Failed";
            } else {
                op_status = "Not started";
            }
        }

        if is_forged {
            forgedCount += 1;
        }

        can_optimize_action = (
            is_forged
            or display_best_raw != None
            or (op_queue_summary and "has_gen_done" in op_queue_summary and op_queue_summary["has_gen_done"])
            or (op_queue_summary and "has_opt_done" in op_queue_summary and op_queue_summary["has_opt_done"])
        );

        has_row_queue_lock = False;
        for locked_op_name in locked_op_names {
            if locked_op_name and op_matches(locked_op_name, op_name) {
                has_row_queue_lock = True;
                break;
            }
        }

        operatorRows.push({
            "name": op_name,
            "short": op_short,
            "baseline": baseline_ms,
            "best": best_ms,
            "speedup": speedup,
            "status": op_status,
            "ratio": ratio_for_sort,
            "is_forged": is_forged,
            "can_optimize_action": can_optimize_action,
            "action_disabled": has_row_queue_lock,
            "call_count": op_count_n,
            "baseline_raw": b_val_raw,
            "display_order": get_stable_op_display_order(op_name)
        });
    }

    grandTotalPytorchTime = 0.0;
    for r in operatorRows {
        c_count = r["call_count"] if "call_count" in r else 1;
        b_raw = r["baseline_raw"] if "baseline_raw" in r else None;
        if b_raw != None and b_raw > 0 {
            grandTotalPytorchTime += b_raw * c_count;
        }
    }

    for r in operatorRows {
        r["runtime_pct"] = "--";
        c_count = r["call_count"] if "call_count" in r else 1;
        b_raw = r["baseline_raw"] if "baseline_raw" in r else None;
        if b_raw != None and b_raw > 0 and grandTotalPytorchTime > 0 {
            pct = ((b_raw * c_count) / grandTotalPytorchTime) * 100.0;
            if pct < 0.1 {
                r["runtime_pct"] = "<0.1%";
            } else {
                r["runtime_pct"] = f"{pct.toFixed(2)}%";
            }
        }
    }

    modelSpeedupRatio = 0.0;
    if totalAllPytorchMs > 0 and totalPytorchMs > totalForgedMs {
        effectiveMs = totalAllPytorchMs - (totalPytorchMs - totalForgedMs);
        if effectiveMs > 0 {
            modelSpeedupRatio = totalAllPytorchMs / effectiveMs;
        }
    }

    projectBackend = status["backend"] if status and "backend" in status else "CUDA";
    metricsDefaultMode = (
        str(metricsMeta["default_mode"])
        if metricsMeta and "default_mode" in metricsMeta and metricsMeta["default_mode"]
        else "deployment"
    );
    metricsSelectionPolicy = (
        str(metricsMeta["selection_policy"])
        if metricsMeta and "selection_policy" in metricsMeta and metricsMeta["selection_policy"]
        else "safe"
    );
    speedupLabel = (
        str(metricsMeta["speedup_label"])
        if metricsMeta and "speedup_label" in metricsMeta and metricsMeta["speedup_label"]
        else "Estimated Op-Level Speedup"
    );

    return
        <div className="p-6 md:p-8 max-w-7xl mx-auto space-y-6">
            <DashboardHeader
                name={name}
                forgedCount={forgedCount}
                opsCount={list_len(to_list(ops))}
                modelSpeedupRatio={modelSpeedupRatio}
                speedupLabel={speedupLabel}
                benchmarkMode={metricsDefaultMode}
                selectionPolicy={metricsSelectionPolicy}
                hardwareStatus={hardwareStatus}
                projectBackend={projectBackend}
                forgedTooltipPos={forgedTooltipPos}
                setForgedTooltipPos={setForgedTooltipPos}
            />
            {<GpuMismatchBanner
                benchmarkedGpuName={benchmarkedGpuName}
                currentGpuName={gpu_name}
                onClone={handleCloneForGpu}
                onDismiss={handleDismissGpuMismatch}
                cloneBusy={gpuCloneBusy}
            /> if gpuMismatch and not gpuMismatchDismissed else <span/>}
            <div className="space-y-4">
                <WorkQueuePanel
                    name={name}
                    activeTasks={active_tasks_for_queue}
                    activePoll={active_poll}
                    isStarting={postSubmitActive}
                    startingTag={startingHint["tag"]}
                    startingOp={startingHint["op"]}
                    startingWorkers={startingHint["workers"]}
                    currentOperator={current_operator}
                    benchmarkSlot={benchmark_slot}
                    pendingOperators={pending_operators}
                    queuedJobs={queued_jobs}
                    speedData={speedData}
                    mctsSummary={mctsSummary}
                    onClearQueue={handleClearQueue}
                />
                <OperatorResultsTable
                    operatorRows={operatorRows}
                    name={name}
                    navigate={navigate}
                    onOpenActionModal={openActionModal}
                    onOpenBenchLog={lambda -> None { setBenchLogOpen(True);}}
                    loading={loading}
                    profiling={profile_active}
                    gpuMismatch={gpuMismatch}
                />
            </div>
            {<GenerateModal
                key={"generate-modal-" + str(generateModalSession)}
                isOpen={generateModalOpen}
                modalSession={generateModalSession}
                onClose={closeGenerateModal}
                onComplete={closeGenerateModal}
                onStarted={handleModalStarted}
                hasProjectWork={any_job_active}
                projectName={name}
                opName={modalOpName}
            />
            if generateModalOpen
            else <span/>}
            {<OptimizeModal
                key={"optimize-modal-" + str(optimizeModalSession)}
                isOpen={optimizeModalOpen}
                modalSession={optimizeModalSession}
                onClose={closeOptimizeModal}
                onComplete={closeOptimizeModal}
                onStarted={handleModalStarted}
                hasProjectWork={any_job_active}
                projectName={name}
                opName={modalOpName}
            />
            if optimizeModalOpen
            else <span/>}
            <GpuMismatchModal
                isOpen={gpuMismatchModalOpen}
                onClose={lambda -> None { setGpuMismatchModalOpen(False);}}
                benchmarkedGpuName={benchmarkedGpuName}
                currentGpuName={gpu_name}
                onClone={handleCloneForGpu}
                onReturnToProject={lambda -> None { setGpuMismatchModalOpen(False);}}
                cloneBusy={gpuCloneBusy}
            />
            <BenchmarkLogModal
                isOpen={benchLogOpen}
                name={name}
                onClose={lambda -> None { setBenchLogOpen(False);}}
                benchLog={benchLog}
                benchLogLoading={benchLogLoading}
                benchLogCopied={benchLogCopied}
                onRefresh={refreshBenchLog}
                onCopy={copyBenchLog}
            />
            <JobLogModal
                isOpen={logModalOpen}
                onClose={lambda -> None { setLogModalOpen(False);}}
                activeLogKey={activeLogKey}
                logTail={logTail}
            />
        </div>;
}
