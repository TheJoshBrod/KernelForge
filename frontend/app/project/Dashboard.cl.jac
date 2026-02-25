"""Dashboard Page"""

import from react { useEffect, useLayoutEffect, useState }
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
import from .shared.MetricsAdapter {
    to_num,
    to_list,
    list_len,
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

    has ops: list = [],
        loading: bool = True,
        status: dict = {},
        hardwareInfo: dict = {},
        hardwareLoading: bool = False,
        mctsSummary: dict = {},
        benchLogOpen: bool = False,
        benchLogLoading: bool = False,
        benchLogCopied: bool = False,
        benchLog: str = "",
        logTail: list = [],
        activeLogKey: str = "",
        logModalOpen: bool = False;

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
        loading = True;
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
                # Process Ops
                ops_data = ops_resp.reports[0]
                if ops_resp and ops_resp.reports
                else None;
                ops = to_list(ops_data["ops"])
                if ops_data and "ops" in ops_data
                else [];
                # Process Metrics (same source used by Operator Workbench)
                metrics_data = metrics_resp.reports[0]
                if metrics_resp and metrics_resp.reports
                else {};
                if metrics_data {
                    if "operator_usage" in metrics_data {
                        ops = to_list(metrics_data["operator_usage"]);
                    }
                    setSpeedData(get_metrics_speed_source(metrics_data));
                }
                # Process Status
                status_data = status_resp.reports[0]
                if status_resp and status_resp.reports
                else {};
                status = status_data or {};
                # Process Hardware
                hw_data = hw_resp.reports[0] if hw_resp and hw_resp.reports else {};
                hardwareInfo = hw_data or {};
                # Process MCTS
                mcts_data = mcts_resp.reports[0]
                if mcts_resp and mcts_resp.reports
                else {};
                mctsSummary = mcts_data or {};
                persist_dashboard_cache(name);
            }
        } except Exception as e {
            console.error("Failed to fetch dashboard data:", e);
        } finally {
            loading = False;
        }
    }

    useLayoutEffect(
        lambda -> None { if name {
            cached = read_dashboard_cache(name);
            if cached {
                if "operator_usage" in cached {
                    ops = to_list(cached["operator_usage"]);
                }
                if "speed_comparison" in cached {
                    setSpeedData(to_list(cached["speed_comparison"]));
                }
                if "status" in cached and cached["status"] {
                    status = cached["status"];
                }
                if "hardware_info" in cached and cached["hardware_info"] {
                    hardwareInfo = cached["hardware_info"];
                }
                if "mcts_summary" in cached and cached["mcts_summary"] {
                    mctsSummary = cached["mcts_summary"];
                }
                loading = False;
            }
        }},
        [name]
    );

    useEffect(lambda -> None { refreshData();}, [name]);

    # Auto-refresh status (faster while active, slower while idle)
    useEffect(
        lambda -> Any { poll_ms = get_profile_poll_ms();timer = setInterval(
            lambda -> None { __jacSpawn("GetProjectStatus", "", {"projectName": name}).then(
                lambda resp: any  -> None { data = resp.reports[0]
                if resp and resp.reports
                else {};status = data or {};
                __jacSpawn("GetProjectMetrics", "", {"projectName": name}).then(
                    lambda c_resp: any  -> None { c_data = c_resp.reports[0]
                    if c_resp and c_resp.reports
                    else {};
                    if c_data {
                        if "operator_usage" in c_data {
                            ops = to_list(c_data["operator_usage"]);
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
                    mctsSummary = m_data or {};
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
                    activeLogKey = log_job_key;
                    __jacSpawn("GetJobLog", "", {"projectName": name, "jobKey": log_job_key, "lines": 12}).then(
                        lambda l_resp: any -> None { l_data = l_resp.reports[0]
                        if l_resp and l_resp.reports
                        else {};
                        logTail = l_data["lines"] if l_data and "lines" in l_data else [];}
                    ).catch(lambda e: any -> None {});
                } else {
                    logTail = [];
                    activeLogKey = "";
                }}
            ).catch(
                lambda e: any  -> None {
                # console.error("Failed to poll project status:", e);
                }
            );},
            poll_ms
        );return lambda -> None { clearInterval(timer);}; },
        [name, forgeBusy, status]
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
        benchLogLoading = True;
        benchLogCopied = False;
        try {
            rel_path = f"kernels/projects/{name}/logs/benchmark.log";
            resp = await __jacSpawn(
                "ReadRepoFile", "", {"projectName": name, "relPath": rel_path}
            );
            data = resp.reports[0] if resp and resp.reports else {};
            if data and "success" in data and data["success"] {
                benchLog = data["content"] if "content" in data else "";
            } else {
                benchLog = data["error"] if data and "error" in data else "No log yet.";
            }
        } except Exception as e {
            benchLog = f"Failed to load benchmark log: {e}";
        } finally {
            benchLogLoading = False;
        }
    }

    def copyBenchLog -> None {
        if not benchLog {
            return;
        }
        navigator.clipboard.writeText(benchLog).then(
            lambda -> None { benchLogCopied = True;setTimeout(
                lambda -> None { benchLogCopied = False;}, 1500
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

    current_state_meta = "";
    if current_started_at and current_updated_at {
        current_state_meta = f"Started: {current_started_at}  |  Updated: {current_updated_at}";
    } elif current_started_at {
        current_state_meta = f"Started: {current_started_at}";
    }

    forgeStateToneClass = "border-[var(--border)] bg-[var(--surface-0)] text-zinc-300";
    if current_phase == "completed" {
        forgeStateToneClass = "border-emerald-500/45 bg-emerald-500/10 text-emerald-200";
    } elif current_phase == "completed_with_failures" {
        forgeStateToneClass = "border-yellow-500/45 bg-yellow-500/10 text-yellow-200";
    } elif current_phase == "failed" {
        forgeStateToneClass = "border-red-500/45 bg-red-500/10 text-red-200";
    } elif active_poll or current_phase == "queued" {
        forgeStateToneClass = "border-[var(--accent-orange)]/45 bg-[var(--accent-orange)]/10 text-orange-100";
    }

    nodesExplored = 0;
    if mctsSummary and "total_nodes" in mctsSummary and mctsSummary["total_nodes"] != None {
        nodesExplored = parseInt(f"{mctsSummary['total_nodes']}");
    } elif mctsSummary and "ops" in mctsSummary {
        nodesExplored = list_len(to_list(mctsSummary["ops"]));
    } else {
        nodesExplored = list_len(ops);
    }
    if nodesExplored != nodesExplored or nodesExplored < 0 {
        nodesExplored = 0;
    }
    if nodesExplored == 0 and list_len(speedData) > 0 {
        nodesExplored = list_len(speedData);
    }

    kpis = compute_speed_kpis(speedData, ops);
    bestSpeedup = to_num(kpis["best_speedup"] if kpis and "best_speedup" in kpis else 1);
    if bestSpeedup is None or bestSpeedup <= 0 {
        bestSpeedup = 1;
    }
    hasBestSpeedup = (
        True
        if kpis and "has_best_speedup" in kpis and kpis["has_best_speedup"]
        else False
    );
    totalSavedMs = to_num(kpis["total_saved_ms"] if kpis and "total_saved_ms" in kpis else 0);
    if totalSavedMs is None or totalSavedMs < 0 {
        totalSavedMs = 0;
    }
    bestSpeedupText = f"{bestSpeedup.toFixed(2)}x" if hasBestSpeedup else "--";
    totalSavedText = f"{totalSavedMs.toFixed(2)} ms" if totalSavedMs > 0 else "0.00 ms";

    metricCardClass = "relative clip-anvil border border-[var(--border)] bg-[color:rgba(15,18,24,0.76)] backdrop-blur-md p-4 shadow-[0_14px_30px_rgba(0,0,0,0.38)] overflow-hidden";

    status_cards = [];
    status_cards.push(
        <div key="metric-nodes" className={metricCardClass}>
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-[var(--accent-orange)]/65 to-transparent"></div>
            <div className="flex items-center justify-between mb-3">
                <span className="text-[10px] uppercase tracking-[0.16em] text-zinc-500">
                    Nodes Explored
                </span>
                <ShieldCheckIcon className="w-4 h-4 text-[var(--accent-orange)]/90"/>
            </div>
            <div className="text-2xl font-semibold text-[var(--text-strong)]">
                {nodesExplored}
            </div>
            <div className="mt-1 text-[11px] text-zinc-500">
                MCTS operator tree nodes
            </div>
        </div>
    );

    status_cards.push(
        <div key="metric-speedup" className={metricCardClass}>
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-[var(--accent-orange)]/65 to-transparent"></div>
            <div className="flex items-center justify-between mb-3">
                <span className="text-[10px] uppercase tracking-[0.16em] text-zinc-500">
                    Best Speedup
                </span>
                <ArrowRightIcon className="w-4 h-4 text-[var(--accent-orange)]/90"/>
            </div>
            <div className="text-2xl font-semibold text-[var(--accent-orange)]">
                {bestSpeedupText}
            </div>
            <div className="mt-1 text-[11px] text-zinc-500">
                Kernel Forge vs PyTorch
            </div>
        </div>
    );
    status_cards.push(
        <div key="metric-saved" className={metricCardClass}>
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-[var(--accent-orange)]/65 to-transparent"></div>
            <div className="flex items-center justify-between mb-3">
                <span className="text-[10px] uppercase tracking-[0.16em] text-zinc-500">
                    Total Time Saved
                </span>
                <CheckIcon className="w-4 h-4 text-[var(--accent-orange)]/90"/>
            </div>
            <div className="text-xl font-semibold text-[var(--text-strong)]">
                {totalSavedText}
            </div>
            <div className="mt-1 text-[11px] text-zinc-500">
                Estimated using operator counts
            </div>
        </div>
    );

    charts_empty_message = "No benchmark results yet. Run Benchmark to see PyTorch timings.";
    if profile_active {
        charts_empty_message = "Profiling model...";
    } elif gen_active and gen_phase == "benchmarking" {
        charts_empty_message = "Benchmarking generated kernels...";
    } elif gen_active {
        charts_empty_message = "Generating kernels...";
    } elif opt_active {
        charts_empty_message = "Optimizing kernels...";
    } elif bench_raw == "queued" {
        charts_empty_message = "Benchmark queued...";
    } elif bench_raw == "running" {
        charts_empty_message = (
            f"{bench_message}{bench_progress_text}"
            if bench_message
            else f"Benchmarking PyTorch baseline...{bench_progress_text}"
        );
    } elif bench_raw == "error" or bench_raw == "failed" {
        charts_empty_message = "Benchmark failed (see benchmark.log).";
    } elif bench_raw == "success" or bench_raw == "completed" {
        charts_empty_message = "No benchmark data available.";
    }

    manualSelectedMap = {};
    for op_name in manualOps {
        manualSelectedMap[str(op_name)] = True;
    }
    manualOpItems = [];
    for op in to_list(ops) {
        if not op or not ("name" in op) or not op["name"] {
            continue;
        }
        op_name = str(op["name"]);
        checked = op_name in manualSelectedMap;
        manualOpItems.push(
            <label
                key={op_name}
                className="flex items-center gap-2 text-xs text-zinc-300 py-0.5"
            >
                <input
                    type="checkbox"
                    checked={checked}
                    onChange={lambda e: any  -> None { toggleManualOp(op_name);}}
                    className="w-3.5 h-3.5 rounded border-zinc-700 bg-zinc-900 text-amber-500 focus:ring-amber-500/50"
                />
                <span className="truncate">
                    {format_op_short(op_name)}
                </span>
            </label>
        );
    }
    manualSelectedCount = 0;
    for op_name in manualOps {
        if str(op_name) in manualSelectedMap {
            manualSelectedCount += 1;
        }
    }

    return
        <div className="p-6 md:p-8 max-w-7xl mx-auto space-y-6">
            <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
                <div className="space-y-1">
                    <h2 className="text-2xl md:text-3xl font-semibold tracking-tight text-[var(--text-strong)]">
                        Project Dashboard
                    </h2>
                    <p className="text-[10px] uppercase tracking-[0.22em] text-[var(--text-muted)]">
                        Forge command center · {name}
                    </p>
                </div>
                <div className="flex items-center gap-3 bg-[var(--surface-0)] border border-[var(--border)] rounded-lg p-2.5 shrink-0 shadow-sm">
                    <div className="flex items-center justify-center w-9 h-9 rounded bg-zinc-900/60 border border-[var(--border)]">
                        <CpuIcon className="w-4 h-4 text-zinc-400"/>
                    </div>
                    <div className="pr-2">
                        <div className="text-[9px] uppercase tracking-[0.16em] text-zinc-500 mb-0.5 flex items-center gap-2">
                            <span>Selected GPU</span>
                            <span className="w-1 h-1 rounded-full bg-zinc-700"></span>
                            <span>{activeBackend}</span>
                        </div>
                        <div className="text-sm font-semibold text-[var(--text-strong)] truncate max-w-[280px]" title={hardwareStatus}>
                            {hardwareStatus}
                        </div>
                    </div>
                </div>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <div className="clip-anvil border border-[var(--border)] bg-[var(--surface-0)] p-4 space-y-3">
                    <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-zinc-200">
                        Forge Control
                    </h3>
                    <div className="flex items-center gap-2">
                        <GlowButton
                            label={(
                                "Heating forge..."
                                if forgeBusy
                                else "Start / Resume Forging"
                            )}
                            onClick={lambda -> None { startForge();}}
                            disabled={forgeBusy
                            or not name
                            or profile_active
                            or gen_active
                            or opt_active}
                        />
                        <button
                            className="px-3 py-2 rounded-lg border border-[var(--accent-orange)]/35 bg-[var(--accent-orange)]/10 text-orange-100 text-xs font-semibold hover:bg-[var(--accent-orange)]/20 transition-colors inline-flex items-center gap-1"
                            onClick={lambda -> None { openWorkbench();}}
                        >
                            Open Workbench
                            <ArrowRightIcon className="w-3 h-3"/>
                        </button>
                    </div>
                    {<div className="text-xs text-red-300">
                        {forgeError}
                    </div>
                    if forgeError
                    else <div className="text-xs text-zinc-400">
                        {forgeNotice}
                    </div>
                    if forgeNotice
                    else <span></span>}
                </div>
                <div className="clip-anvil border border-[var(--border)] bg-[var(--surface-0)] p-4 space-y-3">
                    <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-zinc-200">
                        Run Mode
                    </h3>
                    <div className="inline-flex rounded-md border border-[var(--border)] overflow-hidden bg-[var(--surface-1)]">
                        <button
                            className={"px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.15em] transition-colors " + (
                                "bg-[var(--accent-orange)]/15 text-orange-100"
                                if forgeMode == "auto"
                                else "text-zinc-400 hover:text-zinc-200"
                            )}
                            onClick={lambda -> None { setForgeMode("auto");}}
                        >
                            Automatic
                        </button>
                        <button
                            className={"px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.15em] border-l border-[var(--border)] transition-colors " + (
                                "bg-[var(--accent-orange)]/15 text-orange-100"
                                if forgeMode == "manual"
                                else "text-zinc-400 hover:text-zinc-200"
                            )}
                            onClick={lambda -> None { setForgeMode("manual");}}
                        >
                            Manual
                        </button>
                    </div>
                    {<div className="space-y-2">
                        <div className="flex items-center gap-2">
                            <button
                                className="px-2 py-1 rounded-md text-[10px] font-semibold uppercase tracking-[0.14em] border border-[var(--border)] bg-[var(--surface-1)] text-zinc-300"
                                onClick={lambda -> None { selectAllManualOps();}}
                            >
                                Select all
                            </button>
                            <button
                                className="px-2 py-1 rounded-md text-[10px] font-semibold uppercase tracking-[0.14em] border border-[var(--border)] bg-[var(--surface-1)] text-zinc-300"
                                onClick={lambda -> None { clearManualOps();}}
                            >
                                Clear
                            </button>
                        </div>
                        <div className="max-h-28 overflow-y-auto clip-anvil border border-[var(--border)] bg-[var(--surface-1)] p-2 space-y-1">
                            {manualOpItems
                            if list_len(manualOpItems) > 0
                            else <div className="text-xs text-zinc-500">
                                No operators discovered yet.
                            </div>}
                        </div>
                        <div className="text-[11px] uppercase tracking-[0.14em] text-zinc-500">
                            Selected: {manualSelectedCount}
                        </div>
                    </div>
                    if forgeMode == "manual"
                    else <div className="text-xs text-zinc-400">
                        Automatic mode processes all discovered operators.
                    </div>}
                </div>
                <div className="clip-anvil border border-[var(--border)] bg-[var(--surface-0)] p-4 space-y-3">
                    <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-zinc-200">
                        Forge State
                    </h3>
                    <div
                        className={"px-3 py-2 clip-anvil border text-xs font-semibold uppercase tracking-[0.16em] " + forgeStateToneClass}
                    >
                        {forge_status}
                    </div>
                    <div className="text-sm text-zinc-200 leading-relaxed">
                        {forge_detail}
                    </div>
                    {<div className="w-full h-1.5 rounded-sm bg-zinc-900/80 overflow-hidden border border-[var(--border)]">
                        <div
                            className="forge-progress-fill"
                            style={{"width": f"{current_progress_bar}%"}}
                        ></div>
                    </div>
                    if active_poll
                    else <span></span>}
                    {<div
                        className="mt-1 relative rounded border border-[var(--border)] bg-zinc-950/60 overflow-hidden cursor-pointer group"
                        onClick={lambda -> None { logModalOpen = True;}}
                    >
                        <div className="p-2 font-mono text-[10px] text-zinc-400 leading-[1.6] max-h-[4.5rem] overflow-hidden">
                            <div className="text-[9px] uppercase tracking-[0.14em] text-zinc-600 mb-1">
                                {activeLogKey + ".log — live tail"}
                            </div>
                            {log_line_nodes}
                        </div>
                        <div className="absolute inset-0 flex items-center justify-center bg-zinc-950/75 opacity-0 group-hover:opacity-100 transition-opacity duration-150 pointer-events-none">
                            <span className="text-[11px] font-semibold uppercase tracking-[0.14em] text-zinc-100">
                                View more details →
                            </span>
                        </div>
                    </div>
                    if active_poll and logTail.length > 0
                    else <span></span>}
                    <div className="text-[10px] uppercase tracking-[0.16em] text-zinc-500">
                        Auto-refresh: {poll_label}
                    </div>
                    {<div className="text-[10px] text-zinc-500 leading-relaxed">
                        {current_state_meta}
                    </div>
                    if current_state_meta
                    else <span></span>}
                </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {status_cards}
            </div>
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
                <div className="xl:col-span-2 clip-anvil border border-[var(--border)] bg-[var(--surface-0)] p-4 space-y-3">
                    <div className="flex flex-wrap justify-between items-center gap-3">
                        <div>
                            <h3 className="text-sm font-semibold text-zinc-200">
                                Speed Comparison
                            </h3>
                            <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-500">
                                Top 8 slowest operators by baseline time
                            </p>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="inline-flex rounded-md border border-[var(--border)] overflow-hidden bg-[var(--surface-1)]">
                                <button
                                    className={"px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] transition-colors " + (
                                        "bg-[var(--accent-orange)]/15 text-orange-100"
                                        if comparisonView == "ratio"
                                        else "text-zinc-400 hover:text-zinc-200"
                                    )}
                                    onClick={lambda -> None { setComparisonView("ratio");}}
                                >
                                    Ratio
                                </button>
                                <button
                                    className={"px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] border-l border-[var(--border)] transition-colors " + (
                                        "bg-[var(--accent-orange)]/15 text-orange-100"
                                        if comparisonView == "absolute"
                                        else "text-zinc-400 hover:text-zinc-200"
                                    )}
                                    onClick={lambda -> None { setComparisonView("absolute");}}
                                >
                                    Absolute ms
                                </button>
                            </div>
                            <button
                                onClick={lambda -> None { benchLogOpen = True;}}
                                className="text-[10px] px-2 py-1 rounded border border-[var(--border)] bg-[var(--surface-1)] text-zinc-300"
                            >
                                benchmark.log
                            </button>
                        </div>
                    </div>
                    <div className="text-[11px] text-zinc-400">
                        Mode: {effectiveComparisonView}
                    </div>
                    {<div className="w-full h-[250px] min-h-[250px]">
                        <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={0}>
                            <BarChart
                                data={to_list(speedChartData)}
                                margin={{"top": 8, "right": 12, "left": 12, "bottom": 8}}
                            >
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.18)" vertical={False}/>
                                <XAxis dataKey="name" stroke="#7a8494" fontSize={11} tickLine={False} axisLine={False}/>
                                {<YAxis
                                    stroke="#7a8494"
                                    fontSize={11}
                                    tickLine={False}
                                    axisLine={False}
                                    tickFormatter={format_tick_ratio}
                                />
                                if effectiveComparisonView == "ratio"
                                else <YAxis
                                    stroke="#7a8494"
                                    fontSize={11}
                                    tickLine={False}
                                    axisLine={False}
                                    tickFormatter={format_tick_ms}
                                />}
                                <Tooltip
                                    cursor={{"fill": 'rgba(20, 26, 34, 0.45)'}}
                                    contentStyle={{
                                        "backgroundColor": 'rgba(11, 16, 22, 0.96)',
                                        "borderColor": 'rgba(120, 132, 148, 0.35)',
                                        "borderRadius": '8px',
                                        "color": '#e5e7eb'
                                    }}
                                    formatter={format_tick_ratio if effectiveComparisonView == "ratio" else format_tick_ms}
                                />
                                <Bar dataKey="ratio" name="Kernel Forge / PyTorch" fill="#e17055" radius={[4, 4, 0, 0]} minPointSize={3} hide={effectiveComparisonView != "ratio"}/>
                                <Bar dataKey="forged" name="Kernel Forge" fill="#e17055" radius={[4, 4, 0, 0]} minPointSize={3} hide={effectiveComparisonView != "absolute"}/>
                                <Bar dataKey="pytorch" name="PyTorch" fill="#94a3b8" radius={[4, 4, 0, 0]} minPointSize={3} hide={effectiveComparisonView != "absolute"}/>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                    if hasSpeedChart
                    else <div className="text-zinc-500 text-sm italic">
                        {charts_empty_message}
                    </div>}
                </div>
                <div className="clip-anvil border border-[var(--border)] bg-[var(--surface-0)] p-4 space-y-3">
                    <div className="flex justify-between items-center gap-2">
                        <h3 className="text-sm font-semibold text-zinc-200">
                            Operator Time Consumption
                        </h3>
                        <div className="inline-flex rounded-md border border-[var(--border)] overflow-hidden bg-[var(--surface-1)]">
                            <button
                                className={"px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] transition-colors " + (
                                    "bg-[var(--accent-orange)]/15 text-orange-100"
                                    if opTimeMode == "best"
                                    else "text-zinc-400 hover:text-zinc-200"
                                )}
                                onClick={lambda -> None { setOpTimeMode("best");}}
                            >
                                Best
                            </button>
                            <button
                                className={"px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] border-l border-[var(--border)] transition-colors " + (
                                    "bg-[var(--accent-orange)]/15 text-orange-100"
                                    if opTimeMode == "forged"
                                    else "text-zinc-400 hover:text-zinc-200"
                                )}
                                onClick={lambda -> None { setOpTimeMode("forged");}}
                            >
                                Forged
                            </button>
                            <button
                                className={"px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] border-l border-[var(--border)] transition-colors " + (
                                    "bg-[var(--accent-orange)]/15 text-orange-100"
                                    if opTimeMode == "pytorch"
                                    else "text-zinc-400 hover:text-zinc-200"
                                )}
                                onClick={lambda -> None { setOpTimeMode("pytorch");}}
                            >
                                PyTorch
                            </button>
                        </div>
                    </div>
                    {<div className="space-y-2 max-h-[320px] overflow-y-auto pr-1">
                        {to_list(timeListRows).map(
                            lambda item: dict  -> Any { return <div
                                key={f"{item['rank']}-{item['name']}"}
                                className="clip-anvil border border-[var(--border)] bg-[var(--surface-1)] px-3 py-2"
                            >
                                <div className="flex items-center justify-between gap-2">
                                    <span className="text-xs text-zinc-200 truncate">
                                        #{item["rank"] if item and "rank" in item else ""} {item["name"] if item and "name" in item else ""}
                                    </span>
                                    <span className="text-[11px] font-mono text-zinc-300 shrink-0">
                                        {format_tick_ms(item["value"] if item and "value" in item else 0)}
                                    </span>
                                </div>
                                <div className="mt-2 h-1.5 rounded-full border border-[var(--border)] bg-zinc-900/80 overflow-hidden">
                                    <div
                                        className="h-full rounded-full bg-gradient-to-r from-[var(--accent-orange)] to-[var(--molten-gold)] transition-all duration-500 ease-out"
                                        style={{"width": item["bar_width"] if item and "bar_width" in item else "0%"}}
                                    ></div>
                                </div>
                            </div>; }
                        )}
                    </div>
                    if list_len(timeListRows) > 0
                    else <div className="text-zinc-500 text-sm italic">
                        {charts_empty_message}
                    </div>}
                </div>
            </div>
            <Modal
                isOpen={benchLogOpen}
                onClose={lambda -> None { benchLogOpen = False;}}
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
                onClose={lambda -> None { logModalOpen = False;}}
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
        </div>;
}
