"""Dashboard Page"""

import from react { useEffect, useLayoutEffect, useState }
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

    post_submit_state = useState(False);
    postSubmitActive = post_submit_state[0];
    setPostSubmitActive = post_submit_state[1];

    starting_hint_state = useState({"tag": "", "op": "", "workers": 1});
    startingHint = starting_hint_state[0];
    setStartingHint = starting_hint_state[1];

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

    async def handleModalStarted(hint: any) -> None {
        setQueueManuallyCleared(False);
        setGenerateModalOpen(False);
        setOptimizeModalOpen(False);
        hint_tag = hint["_tag"] if hint and "_tag" in hint else "";
        hint_op = hint["_op"] if hint and "_op" in hint else "";
        hint_workers = hint["_workers"] if hint and "_workers" in hint else 1;
        setStartingHint({"tag": hint_tag, "op": hint_op, "workers": hint_workers});
        setPostSubmitActive(True);
        # Persist across navigation (Dashboard unmounts/remounts when switching tabs).
        window["__kf_post_submit"] = {"tag": hint_tag, "op": hint_op, "workers": hint_workers, "expiry": Date.now() + 120000};
        # Hard fallback: cancel after 120s if tasks never appear (e.g. job errored).
        # The useEffect below cancels it earlier once real tasks are visible.
        setTimeout(lambda -> None { setPostSubmitActive(False); window["__kf_post_submit"] = None;}, 120000);
        # Kick off a single fast status RPC so polling doesn't have to wait
        # a full interval before showing the first real task.
        status_resp = await __jacSpawn("GetProjectStatus", "", {"projectName": name});
        status_data = status_resp.reports[0] if status_resp and status_resp.reports else {};
        if status_data {
            setStatus(status_data);
        }
        refreshData();
    }

    async def handleClearQueue -> None {
        # Suppress the queue display immediately — before the async backend call —
        # so stale poll responses can't cause a blink while we wait for the clear.
        setQueueManuallyCleared(True);
        setPostSubmitActive(False);
        window["__kf_post_submit"] = None;
        # Cancel running jobs and clear backend queue.
        await __jacSpawn("ClearProjectQueue", "", {"projectName": name});
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
        refreshData();
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
            setLoading(False);
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
    _ep_queue_data = status["queue"] if status and "queue" in status else {};
    _ep_active_tasks = _ep_queue_data["active_tasks"] if "active_tasks" in _ep_queue_data else {};
    has_queue_work = Object.keys(_ep_active_tasks).length > 0;
    any_job_active = postSubmitActive or forgeBusy or has_queue_work or is_job_active(_ep_profile) or is_job_active(_ep_gen) or is_job_active(_ep_opt) or is_job_active(_ep_bench);

    # Auto-refresh status (faster while active, slower while idle)
    useEffect(
        lambda -> Any { poll_ms = 500 if any_job_active else 15000;timer = setInterval(
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

    # Restore "starting" indicator if navigated back before tasks appeared.
    useEffect(
        lambda -> None {
            try {
                stored = window["__kf_post_submit"] if "__kf_post_submit" in window else None;
                if stored and stored["expiry"] > Date.now() {
                    setStartingHint({"tag": stored["tag"], "op": stored["op"], "workers": stored["workers"] if "workers" in stored else 1});
                    setPostSubmitActive(True);
                }
            } except Exception {}
        },
        [name]
    );

    # Cancel the "starting" indicator the moment real queue tasks appear.
    useEffect(
        lambda -> None { if postSubmitActive and has_queue_work { setPostSubmitActive(False); setStartingHint({"tag": "", "op": ""}); window["__kf_post_submit"] = None; } },
        [postSubmitActive, has_queue_work]
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
    active_tasks = [];
    for k in Object.keys(active_tasks_dict) {
        t = active_tasks_dict[k];
        t["id"] = k;
        active_tasks.push(t);
    }
    benchmark_slot = queue_data["benchmark_slot"] if "benchmark_slot" in queue_data else {"now": None, "pending": []};
    pending_operators = queue_data["pending_operators"] if "pending_operators" in queue_data else [];
    current_operator = queue_data["current_operator"] if "current_operator" in queue_data else "";
    # While the user has manually cleared the queue, pass empty arrays so
    # WorkQueuePanel shows nothing regardless of stale poll responses.
    active_tasks_for_queue = [] if queueManuallyCleared else active_tasks;

    forgedCount = 0;
    totalPytorchMs = 0.0;
    totalForgedMs = 0.0;
    totalAllPytorchMs = 0.0;
    operatorRows = [];

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

        for sd in to_list(speedData) {
            if sd and "name" in sd and op_matches(str(sd["name"]), op_name) {
                b_val = to_num(sd["pytorch"] if "pytorch" in sd else None);
                b_val_raw = b_val;
                if b_val != None { baseline_ms = format_tick_ms(b_val); }

                f_val = to_num(sd["forged"] if "forged" in sd else None);
                if f_val != None { best_ms = format_tick_ms(f_val); }

                if b_val != None and b_val > 0 {
                    totalAllPytorchMs = totalAllPytorchMs + b_val * op_count_n;
                    if f_val != None and f_val > 0 {
                        r_val = b_val / f_val;
                        speedup = f"{r_val.toFixed(2)}×";
                        ratio_for_sort = r_val;
                        if r_val > 1.001 {
                            is_forged = True;
                            totalPytorchMs = totalPytorchMs + b_val * op_count_n;
                            totalForgedMs = totalForgedMs + f_val * op_count_n;
                        }
                    }
                }
                break;
            }
        }

        op_status = "Not started";
        for task in active_tasks {
            if str(task["op_name"]) == op_name {
                op_status = str(task["current_step"]);
            }
        }

        if op_status == "Not started" {
            if is_forged {
                op_status = "Forged ✓";
            } elif best_ms != "--" {
                op_status = "Benchmarked";
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
            "is_forged": is_forged,
            "call_count": op_count_n,
            "baseline_raw": b_val_raw
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

    operatorRows.sort(lambda a: dict, b: dict -> int {
        if not a and not b { return 0; }
        if not a { return 1; }
        if not b { return -1; }

        a_stat = a["status"] if "status" in a else "Not started";
        b_stat = b["status"] if "status" in b else "Not started";

        a_active = a_stat != "Not started" and a_stat != "Forged ✓" and a_stat != "Benchmarked";
        b_active = b_stat != "Not started" and b_stat != "Forged ✓" and b_stat != "Benchmarked";
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

    return
        <div className="p-6 md:p-8 max-w-7xl mx-auto space-y-6">
            <DashboardHeader
                name={name}
                forgedCount={forgedCount}
                opsCount={list_len(to_list(ops))}
                modelSpeedupRatio={modelSpeedupRatio}
                hardwareStatus={hardwareStatus}
                forgedTooltipPos={forgedTooltipPos}
                setForgedTooltipPos={setForgedTooltipPos}
            />
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
                />
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
