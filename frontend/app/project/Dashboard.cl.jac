"""Dashboard Page"""

import from react { useEffect, useLayoutEffect, useState, useRef }
import from "react-router-dom" { useNavigate, useParams }
import from ...components.ui.Icons {
    CpuIcon,
    ShieldCheckIcon,
    ArrowRightIcon,
    CheckIcon
}
import from ...components.ui.GlowButton { GlowButton }
import from ...components.ui.Modal { Modal }
import from recharts {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer
}
import from ...context.ConfigContext { useConfig }
import from .operator_workbench.menus.GenerateModal { GenerateModal }
import from .operator_workbench.menus.OptimizeModal { OptimizeModal }
import from .shared.MetricsAdapter {
    to_num,
    to_list,
    list_len,
    op_matches,
    get_metrics_speed_source,
    build_dashboard_metrics_view,
    compute_speed_kpis
}

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

    # Separate state for optimistic queue entries (survives server polls)
    optimistic_tasks_state = useState({});
    optimisticTasks = optimistic_tasks_state[0];
    setOptimisticTasks = optimistic_tasks_state[1];

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

    optimize_modal_open_state = useState(False);
    optimizeModalOpen = optimize_modal_open_state[0];
    setOptimizeModalOpen = optimize_modal_open_state[1];

    modal_op_name_state = useState("");
    modalOpName = modal_op_name_state[0];
    setModalOpName = modal_op_name_state[1];

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

    forge_error_state = useState("");
    forgeError = forge_error_state[0];
    setForgeError = forge_error_state[1];

    forged_tooltip_state = useState(None);
    forgedTooltipPos = forged_tooltip_state[0];
    setForgedTooltipPos = forged_tooltip_state[1];

    forge_notice_state = useState("");
    forgeNotice = forge_notice_state[0];
    setForgeNotice = forge_notice_state[1];

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

    def format_op(raw: str) -> str {
        raw_text = raw or "";
        prefix = "torch_nn_functional_";
        if raw_text.startsWith(prefix) {
            return "torch.nn.functional." + raw_text.replace(prefix, "");
        }
        return raw_text.replace("_", ".");
    }

    def format_op_short(raw: str) -> str {
        full = format_op(raw);
        prefix = "torch.nn.functional.";
        if full.startsWith(prefix) {
            return full.replace(prefix, "");
        }
        return full;
    }

    def format_tick_op(value: any) -> str {
        return format_op_short(str(value));
    }

    def format_tick_ms(value: any) -> str {
        num = to_num(value);
        if num is None {
            return "0 ms";
        }
        if num <= 0 {
            return "0 ms";
        }
        if num < 0.000001 {
            return f"{num.toExponential(2)} ms";
        }
        if num < 0.001 {
            return f"{(num * 1000).toFixed(3)} µs";
        }
        if num < 0.01 {
            return f"{num.toFixed(5)} ms";
        }
        if num < 1 {
            return f"{num.toFixed(3)} ms";
        }
        return f"{num.toFixed(2)} ms";
    }

    def format_tick_ratio(value: any) -> str {
        num = to_num(value);
        if num is None {
            return "--";
        }
        return f"{num.toFixed(2)}x";
    }

    def infer_queue_task_tag(task_id: Any, task_data: dict = None) -> str {
        if task_data and "tag" in task_data and task_data["tag"] {
            return str(task_data["tag"]);
        }
        tid = str(task_id if task_id else "");
        if tid == "seq_opt" or tid.startsWith("opt_") {
            return "[OPT]";
        }
        return "[GEN]";
    }

    def queue_attempt_limit(task_id: Any, task_data: dict = None) -> int {
        if task_data and "attempt_max" in task_data and task_data["attempt_max"] {
            parsed_limit = parseInt(str(task_data["attempt_max"]));
            if parsed_limit and parsed_limit > 0 {
                return parsed_limit;
            }
        }
        return 3 if infer_queue_task_tag(task_id, task_data) == "[OPT]" else 8;
    }

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

    def openActionModal(e: any, opName: str, action: str) -> None {
        e.stopPropagation();
        setModalOpName(opName);
        if action == "Optimize" {
            setOptimizeModalOpen(True);
        } else {
            setGenerateModalOpen(True);
        }
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

    async def handleModalStarted(data: Any = None, started_ops: list = None, tag: str = "[GEN]") -> None {
        setGenerateModalOpen(False);
        setOptimizeModalOpen(False);
        # Optimistic UI: inject placeholder tasks into separate state
        if started_ops and started_ops.length > 0 {
            setOptimisticTasks(lambda prev: Any -> Any {
                updated = Object.assign({}, prev);
                now_ts = Date.now();
                for sop in started_ops {
                    task_key = ("seq_opt" if tag == "[OPT]" else "gen_" + str(sop));
                    updated[task_key] = {
                        "tag": tag,
                        "op_name": str(sop),
                        "current_step": ("Starting" if tag == "[OPT]" else "Monitoring"),
                        "attempt_current": 1,
                        "attempt_max": (3 if tag == "[OPT]" else 8),
                        "status": "Starting",
                        "_created_at": now_ts
                    };
                }
                return updated;
            });
        }
        await refreshData();
        setTimeout(lambda -> None { refreshData();}, 1500);
        setTimeout(lambda -> None { refreshData();}, 3500);
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
        refresh_key = "";
        refresh_token = str(Date.now()) + ":" + str(Math.random());
        if name {
            refresh_key = "cgins.dashboard.refresh_token:" + name;
            try {
                localStorage.setItem(refresh_key, refresh_token);
            } except Exception {
                refresh_key = "";
            }
        }
        setLoading(True);
        try {
            if name {
                # Spawn all requests concurrently
                ops_req = __jacSpawn("GetProjectOps", "", {"projectName": name});
                metrics_req = __jacSpawn(
                    "GetProjectMetrics", "", {"projectName": name}
                );
                status_req = __jacSpawn("GetProjectStatus", "", {"projectName": name});
                hw_req = __jacSpawn("DetectHardware", "", {});
                mcts_req = __jacSpawn(
                    "GetProjectMctsSummary", "", {"projectName": name}
                );
                # Await all responses
                ops_resp = await ops_req;
                metrics_resp = await metrics_req;
                status_resp = await status_req;
                hw_resp = await hw_req;
                mcts_resp = await mcts_req;

                latest_refresh_token = refresh_token;
                if refresh_key {
                    try {
                        stored_refresh_token = localStorage.getItem(refresh_key);
                        if stored_refresh_token {
                            latest_refresh_token = stored_refresh_token;
                        }
                    } except Exception {
                        latest_refresh_token = refresh_token;
                    }
                }
                if latest_refresh_token != refresh_token {
                    return;
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
                        setOps(to_list(metrics_data["operator_usage"]));
                    }
                    setSpeedData(get_metrics_speed_source(metrics_data));
                }
                # Process Status
                status_data = status_resp.reports[0]
                if status_resp and status_resp.reports
                else {};
                setStatus(status_data or {});
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
            latest_refresh_token = refresh_token;
            if refresh_key {
                try {
                    stored_refresh_token = localStorage.getItem(refresh_key);
                    if stored_refresh_token {
                        latest_refresh_token = stored_refresh_token;
                    }
                } except Exception {
                    latest_refresh_token = refresh_token;
                }
            }
            if latest_refresh_token == refresh_token {
                setLoading(False);
            }
        }
    }

    useLayoutEffect(
        lambda -> None { if name {
            cached = read_dashboard_cache(name);
            if cached {
                if "operator_usage" in cached {
                    setOps(to_list(cached["operator_usage"]));
                }
                if "speed_comparison" in cached {
                    setSpeedData(to_list(cached["speed_comparison"]));
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
    any_job_active = forgeBusy or is_job_active(_ep_profile) or is_job_active(_ep_gen) or is_job_active(_ep_opt) or is_job_active(_ep_bench);

    # Track previous active state to catch when jobs finish
    prev_active_ref = useRef(any_job_active);
    useEffect(
        lambda -> None {
            was_active = prev_active_ref.current;
            prev_active_ref.current = any_job_active;
            
            # If a job just transitioned from running -> stopped
            if was_active and not any_job_active {
                refreshData();
            }
        },
        [any_job_active]
    );

    # Auto-refresh status (faster while active, slower while idle)
    useEffect(
        lambda -> Any { poll_ms = 1500 if any_job_active else 15000;timer = setInterval(
            lambda -> None { __jacSpawn("GetProjectStatus", "", {"projectName": name}).then(
                lambda resp: any  -> None { data = resp.reports[0]
                if resp and resp.reports
                else {};
                setStatus(data or {});
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
                    }}
                ).catch(
                    lambda e: any  -> None {
                    # console.error("Failed to poll project metrics:", e);
                    }
                );
                __jacSpawn("GetProjectMctsSummary", "", {"projectName": name}).then(
                    lambda m_resp: any  -> None { m_data = m_resp.reports[0]
                    if m_resp and m_resp.reports
                    else {};
                    setMctsSummary(m_data or {});
                    persist_dashboard_cache(name);}
                ).catch(
                    lambda e: any  -> None {
                    # console.error("Failed to poll mcts summary:", e);
                    }
                );
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
                }}
            ).catch(
                lambda e: any  -> None {
                # console.error("Failed to poll project status:", e);
                }
            );},
            poll_ms
        );return lambda -> None { clearInterval(timer);}; },
        [name, forgeBusy, any_job_active]
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

    log_line_nodes = [];
    for log_line in to_list(logTail) {
        ls = str(log_line);
        lsl = ls.toLowerCase();
        lc = "truncate ";
        if lsl.includes("error") or lsl.includes("failed") or lsl.includes("exception") {
            lc = lc + "text-red-400";
        } elif lsl.includes("warning") {
            lc = lc + "text-yellow-400/80";
        } elif lsl.includes("success") or lsl.includes("passed") or lsl.includes("completed") {
            lc = lc + "text-emerald-400/80";
        } elif ls.startsWith("[workflow]") or ls.startsWith("[Remote]") or ls.startsWith("DEBUG:") {
            lc = lc + "text-zinc-300";
        }
        log_line_nodes.push(<div key={ls} className={lc}>{ls}</div>);
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

    queue_data = status["queue"] if status and "queue" in status else {};
    active_tasks_dict = queue_data["active_tasks"] if "active_tasks" in queue_data else {};
    # Build a set of completed task IDs for optimistic cleanup
    completed_task_ids = {};
    completed_tasks_raw = queue_data["completed_tasks"] if "completed_tasks" in queue_data else [];
    for ct in to_list(completed_tasks_raw) {
        if ct and "id" in ct {
            completed_task_ids[ct["id"]] = True;
        }
    }
    # Merge optimistic entries: add any that server doesn't know about yet
    merged_tasks_dict = Object.assign({}, active_tasks_dict);
    stale_optimistic = [];
    now_ms = Date.now();
    for ok in Object.keys(optimisticTasks) {
        opt_entry = optimisticTasks[ok];
        if not opt_entry {
            stale_optimistic.push(ok);
            continue;
        }
        opt_age = now_ms - (opt_entry["_created_at"] if "_created_at" in opt_entry else 0);
        if ok in active_tasks_dict or ok in completed_task_ids or opt_age > 30000 {
            # Server has this task, or entry expired (>30s) — mark as stale
            stale_optimistic.push(ok);
        } else {
            merged_tasks_dict[ok] = opt_entry;
        }
    }
    # Clean up stale optimistic entries (server now has them)
    if stale_optimistic.length > 0 {
        setOptimisticTasks(lambda prev: Any -> Any {
            cleaned = {};
            for pk in Object.keys(prev) {
                if not stale_optimistic.includes(pk) {
                    cleaned[pk] = prev[pk];
                }
            }
            return cleaned;
        });
    }
    active_tasks = [];
    for k in Object.keys(merged_tasks_dict) {
        t = merged_tasks_dict[k];
        if not t {
            continue;
        }
        normalized_task = Object.assign({}, t);
        normalized_task["id"] = k;
        normalized_task["tag"] = infer_queue_task_tag(k, normalized_task);
        if not ("attempt_current" in normalized_task) or not normalized_task["attempt_current"] {
            normalized_task["attempt_current"] = 1;
        }
        if not ("attempt_max" in normalized_task) or not normalized_task["attempt_max"] {
            normalized_task["attempt_max"] = queue_attempt_limit(k, normalized_task);
        }
        active_tasks.push(normalized_task);
    }
    benchmark_slot = queue_data["benchmark_slot"] if "benchmark_slot" in queue_data else {"now": None, "pending": []};
    pending_operators = queue_data["pending_operators"] if "pending_operators" in queue_data else [];
    current_operator = queue_data["current_operator"] if "current_operator" in queue_data else "";
    # Sort: currently active task first, then by op_name
    active_tasks.sort(lambda a: Any, b: Any -> int {
        if not a or not b {
            return 0;
        }
        a_op = a["op_name"] if "op_name" in a else "";
        b_op = b["op_name"] if "op_name" in b else "";
        a_active = 1 if a_op == current_operator else 0;
        b_active = 1 if b_op == current_operator else 0;
        if a_active != b_active {
            return b_active - a_active;
        }
        return 0;
    });

    # --- Completed Tasks (flat activity log, newest first) ---
    completed_tasks_raw = queue_data["completed_tasks"] if "completed_tasks" in queue_data else [];
    completed_list = [];
    for ct in to_list(completed_tasks_raw) {
        if not ct {
            continue;
        }
        ct_op = ct["op_name"] if "op_name" in ct else "";
        if not ct_op or ct_op == "unknown" {
            ct_id = ct["id"] if "id" in ct else "";
            if ct_id and ct_id.startsWith("gen_") {
                ct_op = ct_id.slice(4);
            } elif ct_id and ct_id.startsWith("opt_") {
                ct_op = ct_id.slice(4);
            } elif ct_id == "seq_opt" {
                continue;
            } elif ct_id {
                ct_op = ct_id;
            } else {
                ct_op = "unknown";
            }
        }
        ct_id = ct["id"] if "id" in ct else ct_op;
        completed_list.push(
            Object.assign(
                {},
                ct,
                {
                    "_resolved_op": ct_op,
                    "tag": infer_queue_task_tag(ct_id, ct)
                }
            )
        );
    }
    # Reverse to show newest first
    completed_list.reverse();

    completed_section_nodes = [];
    for ci_idx in range(list_len(completed_list)) {
        ci = completed_list[ci_idx];
        ci_step = ci["current_step"] if "current_step" in ci else "";
        ci_tag = infer_queue_task_tag(ci["id"] if "id" in ci else ci["_resolved_op"], ci);
        ci_op = ci["_resolved_op"];
        ci_result = ci["result"] if "result" in ci else "";
        ci_time = ci["completed_at"] if "completed_at" in ci else "";
        ci_is_done = (ci_step == "Done");
        ci_is_failed = (ci_step == "Failed");

        # Format the result text
        ci_result_text = "";
        if ci_result and ci_result != "completed" {
            ci_result_text = ci_result;
        }


        ci_icon_class = "text-emerald-400" if ci_is_done else "text-red-400";
        ci_icon = "✓" if ci_is_done else "✗";
        ci_border = "border-l-2 border-emerald-500/40" if ci_is_done else "border-l-2 border-red-500/40";

        completed_section_nodes.push(
            <div key={"cl_" + str(ci_idx)} className={"px-4 py-2 bg-[var(--surface-0)] hover:bg-[var(--surface-1)] transition-colors " + ci_border}>
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2.5 font-mono text-xs">
                        <span className={ci_icon_class + " text-sm w-4 text-center"}>{ci_icon}</span>
                        <span className={"font-semibold " + ("text-emerald-400/70" if ci_tag == "[GEN]" else "text-blue-400/70")}>{ci_tag}</span>
                        <span className="text-zinc-300">{format_op_short(ci_op)}</span>
                        {<span className="text-zinc-500 ml-1">{ci_result_text}</span> if ci_result_text else <span></span>}
                    </div>

                </div>
            </div>
        );
    }

    active_tasks_nodes = [];
    for task in active_tasks {
        c_step = task["current_step"] if "current_step" in task and task["current_step"] else "Queued";
        tag = infer_queue_task_tag(task["id"] if "id" in task else "", task);
        
        track_nodes = [];
        steps = ["Queued", "Starting", "Monitoring", "Generating", "Validating", "Benchmarking"] if tag == "[GEN]" else ["Queued", "Starting", "Generating", "Validating", "Profiling"];
        
        c_idx = -1;
        for i in range(list_len(steps)) {
            if steps[i] == c_step {
                c_idx = i;
            }
        }
        
        for i in range(list_len(steps)) {
            s = steps[i];
            # Only show Queued/Starting if we are currently in them, otherwise hide them to keep the UI clean
            if (s == "Queued" or s == "Starting") and c_idx > 1 {
                continue;
            }
            
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
            if tag == "[OPT]" {
                status_label = "Optimized " + (task["result"] if "result" in task else "");
            } else {
                status_label = "Benchmarked " + (task["result"] if "result" in task else "");
            }
        } elif c_step == "Iter Complete" {
            status_label = "Iteration done " + (task["result"] if "result" in task else "");
        } elif c_step == "Iter Failed" {
            status_label = "Iteration failed";
        } elif c_step == "Failed" {
            status_label = "Failed";
        }
        
        bar_w = "0%";
        if c_step == "Starting" { bar_w = "5%"; }
        elif c_step == "Monitoring" { bar_w = "10%"; }
        elif c_step == "Generating" { bar_w = "33%"; }
        elif c_step == "Validating" { bar_w = "55%"; }
        elif c_step == "Benchmarking" { bar_w = "80%"; }
        elif c_step == "Profiling" { bar_w = "80%"; }
        elif c_step == "Iter Complete" or c_step == "Iter Failed" { bar_w = "90%"; }
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
            max_atmpt = str(queue_attempt_limit(task["id"] if "id" in task else "", task));
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

    forgedCount = 0;
    totalPytorchMs = 0.0;
    totalForgedMs = 0.0;
    operatorRows = [];
    # Build lookup sets for operators with success/failure markers on disk
    generated_ops_list = status["generated_ops"] if status and "generated_ops" in status else [];
    generated_ops_set = {};
    for gop in to_list(generated_ops_list) {
        generated_ops_set[str(gop)] = True;
    }
    failed_ops_list = status["failed_ops"] if status and "failed_ops" in status else [];
    failed_ops_set = {};
    for fop in to_list(failed_ops_list) {
        failed_ops_set[str(fop)] = True;
    }

    for op in to_list(ops) {
        if not op or not ("name" in op) or not op["name"] {
            continue;
        }
        op_name = str(op["name"]);
        op_short = format_op_short(op_name);

        baseline_ms = "--";
        best_ms = "--";
        speedup = "--";
        is_forged = False;
        ratio_for_sort = 0.0;

        for sd in to_list(speedData) {
            if sd and "name" in sd and op_matches(str(sd["name"]), op_name) {
                b_val = to_num(sd["pytorch"] if "pytorch" in sd else None);
                if b_val != None { baseline_ms = format_tick_ms(b_val); }

                f_val = to_num(sd["forged"] if "forged" in sd else None);
                if f_val != None { best_ms = format_tick_ms(f_val); }

                if b_val != None and b_val > 0 and f_val != None and f_val > 0 {
                    r_val = b_val / f_val;
                    speedup = f"{r_val.toFixed(2)}×";
                    ratio_for_sort = r_val;
                    if r_val > 1.001 {
                        is_forged = True;
                        op_count_raw = op["count"] if "count" in op else 1;
                        op_count_n = parseInt(f"{op_count_raw}") if op_count_raw else 1;
                        if not op_count_n or op_count_n <= 0 { op_count_n = 1; }
                        totalPytorchMs = totalPytorchMs + b_val * op_count_n;
                        totalForgedMs = totalForgedMs + f_val * op_count_n;
                    }
                }
                break;
            }
        }
        # Fallback: use MCTS tree best_value when benchmarks haven't produced forged data
        if best_ms == "--" and mctsSummary and "ops" in mctsSummary {
            mcts_ops = to_list(mctsSummary["ops"]);
            for ms_op in mcts_ops {
                if ms_op and "op" in ms_op and op_matches(str(ms_op["op"]), op_name) {
                    tree_best = to_num(ms_op["best_value"] if "best_value" in ms_op else None);
                    if tree_best != None and tree_best > 0 {
                        best_ms = format_tick_ms(tree_best);
                        # Compute speedup if we have a baseline
                        b_raw = to_num(baseline_ms) if baseline_ms != "--" else None;
                        if b_raw != None and b_raw > 0 {
                            r_val = b_raw / tree_best;
                            speedup = f"{r_val.toFixed(2)}×";
                            ratio_for_sort = r_val;
                            if r_val > 1.001 {
                                is_forged = True;
                                op_count_raw = op["count"] if "count" in op else 1;
                                op_count_n = parseInt(f"{op_count_raw}") if op_count_raw else 1;
                                if not op_count_n or op_count_n <= 0 { op_count_n = 1; }
                                totalPytorchMs = totalPytorchMs + b_raw * op_count_n;
                                totalForgedMs = totalForgedMs + tree_best * op_count_n;
                            }
                        }
                    }
                    break;
                }
            }
        }
        # Determine baseline status from persistent data (survives clears)
        op_status = "Not started";
        if is_forged {
            op_status = "Forged ✓";
        } elif best_ms != "--" {
            op_status = "Benchmarked";
        } elif generated_ops_set and op_name in generated_ops_set {
            op_status = "Benchmarked";
        } elif failed_ops_set and op_name in failed_ops_set {
            op_status = "Failed";
        }
        # Check completed tasks (newest first) — can override baseline
        for ct in completed_list {
            if str(ct["_resolved_op"]) == op_name {
                step = str(ct["current_step"]);
                ct_tag = infer_queue_task_tag(ct["id"] if "id" in ct else ct["_resolved_op"], ct);
                if step == "Done" {
                    op_status = "Optimized" if ct_tag == "[OPT]" else "Benchmarked";
                } else {
                    op_status = step;
                }
                break;
            }
        }
        # Active tasks override completed ones
        for task in active_tasks {
            if str(task["op_name"]) == op_name {
                step = str(task["current_step"]);
                t_tag = infer_queue_task_tag(task["id"] if "id" in task else task["op_name"], task);
                if step == "Done" or step == "Iter Complete" {
                    op_status = "Optimized" if t_tag == "[OPT]" else "Benchmarked";
                } else {
                    op_status = step;
                }
            }
        }

        if is_forged {
            forgedCount += 1;
        }

        operatorRows.push({
            "name": op_name,
            "short": op_short,
            "baseline": baseline_ms,
            "best": best_ms,
            "speedup": speedup,
            "status": op_status,
            "ratio": ratio_for_sort,
            "is_forged": is_forged
        });
    }

    modelSpeedupPct = 0.0;
    if totalPytorchMs > 0 and totalForgedMs > 0 and totalPytorchMs > totalForgedMs {
        modelSpeedupPct = ((totalPytorchMs - totalForgedMs) / totalPytorchMs) * 100.0;
    }
    
    operatorRows.sort(lambda a: dict, b: dict -> int {
        if not a and not b { return 0; }
        if not a { return 1; }
        if not b { return -1; }

        a_stat = a["status"] if "status" in a else "Not started";
        b_stat = b["status"] if "status" in b else "Not started";

        a_active = a_stat != "Not started" and a_stat != "Forged ✓" and a_stat != "Benchmarked" and a_stat != "Optimized" and a_stat != "Failed";
        b_active = b_stat != "Not started" and b_stat != "Forged ✓" and b_stat != "Benchmarked" and b_stat != "Optimized" and b_stat != "Failed";
        if a_active and not b_active { return -1; }
        if not a_active and b_active { return 1; }

        a_forged = a["is_forged"] if "is_forged" in a else False;
        b_forged = b["is_forged"] if "is_forged" in b else False;
        if a_forged and not b_forged { return -1; }
        if not a_forged and b_forged { return 1; }
        
        a_ratio = a["ratio"] if "ratio" in a else 0.0;
        b_ratio = b["ratio"] if "ratio" in b else 0.0;
        if a_ratio > b_ratio { return -1; }
        if a_ratio < b_ratio { return 1; }
        
        return 0;
    });

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

        row_status_color = "text-zinc-500";
        if r_stat == "Forged ✓" { row_status_color = "text-emerald-400"; }
        elif r_stat == "Benchmarked" { row_status_color = "text-yellow-400"; }
        elif r_stat == "Optimized" { row_status_color = "text-purple-400"; }
        elif r_stat == "Failed" { row_status_color = "text-red-400"; }
        elif r_stat == "Queued" { row_status_color = "text-blue-400"; }
        elif r_stat != "Not started" and r_stat != "Done" { row_status_color = "text-[var(--accent-orange)] animate-pulse"; }
        
        action_text = "Optimize" if r_stat == "Benchmarked" or r_stat == "Forged ✓" or r_stat == "Done" or r_stat == "Optimized" or r_best != "--" else "Generate";
        
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
                <td className={"py-2.5 px-4 font-mono text-xs " + ("text-emerald-400 font-semibold" if r_is_forged else "text-zinc-500")}>{r_speedup}</td>
                <td className="py-2 px-4 font-mono text-xs flex items-center justify-between">
                    <span className={row_status_color}>{r_stat}</span>
                    <button
                        data-opname={r_name}
                        data-action={action_text}
                        onClick={lambda e: any -> None { openActionModal(e, e.currentTarget.getAttribute("data-opname"), e.currentTarget.getAttribute("data-action")); }}
                        className={action_btn_class}
                    >
                        {action_text}
                    </button>
                </td>
            </tr>
        );
    }

    return
        <div className="p-6 md:p-8 max-w-7xl mx-auto space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-[var(--surface-0)] border border-[var(--border)] rounded-lg p-4 shadow-sm">
                <div className="flex items-center gap-4">
                    <div className="flex items-center justify-center w-10 h-10 rounded-md bg-zinc-900 border border-[var(--border)]">
                        <CpuIcon className="w-5 h-5 text-[var(--accent-orange)]"/>
                    </div>
                    <div>
                        <h2 className="text-xl font-semibold tracking-tight text-[var(--text-strong)]">
                            Project Dashboard
                        </h2>
                        <p className="text-[10px] uppercase tracking-[0.22em] text-[var(--text-muted)] mt-0.5">
                            Forge command center · {name}
                        </p>
                    </div>
                </div>
                
                <div className="flex flex-wrap items-center gap-6 md:gap-8 pb-2 md:pb-0">
                    <div
                        className="flex flex-col cursor-help"
                        onMouseMove={lambda e: any -> None { setForgedTooltipPos({"x": e.clientX, "y": e.clientY}); }}
                        onMouseLeave={lambda -> None { setForgedTooltipPos(None); }}
                    >
                        <span className="text-[9px] uppercase tracking-[0.16em] text-zinc-500 mb-1">Coverage</span>
                        <div className="flex items-baseline gap-1.5">
                            <span className="text-sm font-bold text-zinc-200">{forgedCount}</span>
                            <span className="text-xs text-zinc-500">/ {list_len(to_list(ops))} forged</span>
                        </div>
                    </div>
                    <div className="w-px h-8 bg-[var(--border)] hidden sm:block"></div>
                    <div className="flex flex-col">
                        <span className="text-[9px] uppercase tracking-[0.16em] text-zinc-500 mb-1">Model Speedup</span>
                        {<span className="text-sm font-bold text-emerald-400">+{modelSpeedupPct.toFixed(1)}% faster</span>
                        if modelSpeedupPct > 0
                        else <span className="text-sm font-bold text-zinc-500">—</span>}
                    </div>
                    <div className="w-px h-8 bg-[var(--border)] hidden sm:block"></div>
                    <div className="flex flex-col">
                        <span className="text-[9px] uppercase tracking-[0.16em] text-zinc-500 mb-1">
                            Selected GPU
                        </span>
                        <span className="text-xs font-semibold text-zinc-300" title={hardwareStatus}>{hardwareStatus}</span>
                    </div>
                </div>
            </div>

            <div className="space-y-4">
                <div className="rounded-md border border-[var(--border)] bg-[var(--surface-0)] overflow-hidden">
                    <div className="p-3 border-b border-[var(--border)] bg-zinc-900/40 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                        <div className="flex items-center gap-3 shrink-0">
                            <h3 className="text-xs font-semibold uppercase tracking-[0.12em] text-zinc-200">
                                Work Queue
                            </h3>
                            <button
                                onClick={lambda -> None { __jacSpawn("ClearProjectQueue", "", {"projectName": name}).then(lambda _r: any -> None { refreshData();});}}
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
                            <span className="text-zinc-200 font-semibold">{benchmark_slot["now"] if benchmark_slot["now"] else "idle"}</span>
                            {<span className="text-zinc-500">· Next: <span className="text-zinc-400">{", ".join(pending_operators)}</span></span> if list_len(pending_operators) > 0 else <span></span>}
                        </div> if active_poll else <span></span>}
                    </div>
                    <div className="divide-y divide-[var(--border)] max-h-[400px] overflow-y-auto">
                        {active_tasks_nodes
                        if list_len(active_tasks_nodes) > 0
                        else <div className="p-8 text-center text-xs text-zinc-500 font-mono">{"Queue empty" if list_len(completed_section_nodes) == 0 else "No active tasks"}</div>}
                        {<div>
                            <div className="px-4 py-2 bg-zinc-900/60 border-t border-b border-[var(--border)] flex items-center gap-3">
                                <span className="text-[10px] uppercase tracking-[0.12em] text-zinc-500 font-semibold">Completed</span>
                                <button
                                    onClick={lambda -> None {
                                        # Clear on server, then refresh
                                        __jacSpawn("ClearCompletedTasks", "", {"projectName": name}).then(lambda _r: any -> None { refreshData();});
                                        # Optimistically clear local completed_tasks
                                        if status and "queue" in status and "completed_tasks" in status["queue"] {
                                            status["queue"]["completed_tasks"] = [];
                                            setStatus(Object.assign({}, status));
                                        }
                                    }}
                                    className="text-[10px] px-2 py-0.5 rounded border border-zinc-700 bg-zinc-900 hover:border-zinc-500 hover:text-zinc-200 text-zinc-500 transition-colors font-mono"
                                >
                                    clear
                                </button>
                            </div>
                            <div className="divide-y divide-[var(--border)]">
                                {completed_section_nodes}
                            </div>
                        </div> if list_len(completed_section_nodes) > 0 else <span></span>}
                    </div>
                </div>

                <div className="rounded-md border border-[var(--border)] bg-[var(--surface-0)] overflow-hidden">
                    <div className="p-3 border-b border-[var(--border)] bg-zinc-900/40 flex justify-between items-center">
                        <h3 className="text-xs font-semibold uppercase tracking-[0.12em] text-zinc-200">
                            Operator Results
                        </h3>
                        <div className="flex items-center gap-3">
                            <button
                                onClick={lambda -> None { setBenchLogOpen(True);}}
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
                                {operatorRowsNodes}
                            </tbody>
                        </table>
                        {<div className="p-8 text-center text-xs text-zinc-500 font-mono">
                            No operators discovered yet. Profile the model to find operators.
                        </div> if list_len(operatorRowsNodes) == 0 else <span></span>}
                    </div>
                </div>
            </div>
            <GenerateModal
                isOpen={generateModalOpen}
                onClose={lambda -> None { setGenerateModalOpen(False);}}
                onComplete={lambda -> None { setGenerateModalOpen(False);}}
                onStarted={handleModalStarted}
                projectName={name}
                opName={modalOpName}
            />
            <OptimizeModal
                isOpen={optimizeModalOpen}
                onClose={lambda -> None { setOptimizeModalOpen(False);}}
                onComplete={lambda -> None { setOptimizeModalOpen(False);}}
                onStarted={handleModalStarted}
                projectName={name}
                opName={modalOpName}
            />
            <Modal
                isOpen={benchLogOpen}
                onClose={lambda -> None { setBenchLogOpen(False);}}
                title="Benchmark log"
            >
                <div className="p-4 space-y-3">
                    <div className="flex items-center justify-between gap-3">
                        <div className="text-[10px] uppercase tracking-wider font-bold text-zinc-500">
                            projects/{name}/logs/benchmark.log
                        </div>
                        <div className="flex items-center gap-2">
                            <button
                                onClick={lambda -> None { refreshBenchLog();}}
                                className="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 rounded-md text-xs text-zinc-300 border border-zinc-700 transition-colors"
                            >
                                Refresh
                            </button>
                            <button
                                onClick={lambda -> None { copyBenchLog();}}
                                className="px-3 py-1.5 bg-[var(--accent-orange)] hover:bg-[var(--molten-gold-dark)] rounded-md text-xs text-zinc-950 border border-[var(--accent-orange)]/40 transition-colors"
                            >
                                {"Copied" if benchLogCopied else "Copy"}
                            </button>
                        </div>
                    </div>
                    {<div className="text-xs text-zinc-400">
                        Loading...
                    </div>
                    if benchLogLoading
                    else <pre
                        className="text-xs font-mono text-zinc-200 whitespace-pre-wrap break-words bg-zinc-950 border border-zinc-800 rounded-lg p-3"
                    >
                        {benchLog or "No log yet."}
                    </pre>}
                </div>
            </Modal>
            <Modal
                isOpen={logModalOpen}
                onClose={lambda -> None { setLogModalOpen(False);}}
                title={f"{activeLogKey}.log — Live Tail"}
            >
                <div className="p-4 font-mono text-[11px] leading-[1.75] space-y-px">
                    {log_line_nodes
                    if list_len(log_line_nodes) > 0
                    else <div className="text-zinc-500 text-xs">
                        No log output captured yet.
                    </div>}
                </div>
            </Modal>
            {<div
                style={{"position": "fixed", "left": forgedTooltipPos["x"] + 14, "top": forgedTooltipPos["y"] + 14, "zIndex": 9999, "pointerEvents": "none"}}
                className="bg-zinc-900 border border-zinc-700 rounded-md px-3 py-2 text-xs text-zinc-300 shadow-xl max-w-[220px] leading-relaxed"
            >
                Operators with a custom kernel that outperforms PyTorch's built-in implementation.
            </div>
            if forgedTooltipPos
            else <span/>}
        </div>;
}
