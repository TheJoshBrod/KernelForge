"""Dashboard Page"""

import from react { useEffect, useState }
import from "react-router-dom" { useNavigate, useParams }
import from ...components.ui.Icons {
    CpuIcon,
    ShieldCheckIcon,
    ArrowRightIcon,
    CheckIcon,
    AlertCircleIcon
}
import from ...components.ui.GlowButton { GlowButton }
import from ...components.ui.Modal { Modal }
import from recharts {
    PieChart,
    Pie,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer
}
import from ...context.ConfigContext { useConfig }

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
        benchLog: str = "";

    COLORS = [
        '#86e6a9',
        '#5bbbd3',
        '#b7f0d0',
        '#d8f3ff',
        '#c4b5fd',
        '#a78bfa',
        '#818cf8',
        '#60a5fa',
        '#34d399',
        '#f472b6'
    ];

    pieData = [];
    hasPie = False;
    pieTotal = 0;

    op_idx = 0;
    for op in ops {
        slice_val = op["total_time_ms"] if "total_time_ms" in op else 0;
        pieTotal += slice_val;
        pieData.push(
            {
                "name": format_op_short(op["name"]),
                "value": slice_val,
                "fill": COLORS[op_idx % COLORS.length]
            }
        );
        op_idx += 1;
    }
    hasPie = pieData.length > 0 and pieTotal > 0;

    # Speed Comparison Data State
    speedData_state = useState([]);
    speedData = speedData_state[0];
    setSpeedData = speedData_state[1];

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

    hasForged = False;
    for row in speedData {
        forged_raw = row["forged"] if row else None;
        if forged_raw is None {
            forged_raw = row["optimized"] if row else None;
        }
        if forged_raw is None {
            forged_raw = row["kernel_ms"] if row else None;
        }
        forged_val = to_num(forged_raw);
        if forged_val is not None {
            hasForged = True;
            break;
        }
    }

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
        return f"{value}ms";
    }

    def to_num(value: Any) -> Any {
        if value is None {
            return None;
        }
        try {
            parsed = parseFloat(f"{value}");
            if parsed == parsed {
                return parsed;
            }
        } except Exception {
        # no-op
        }
        return None;
    }

    def to_list(items: Any) -> list {
        normalized = [];
        if not items {
            return normalized;
        }
        try {
            if Array.isArray(items) {
                return items;
            }
        } except Exception {
        # no-op
        }
        try {
            for item in items {
                normalized.push(item);
            }
            if normalized.length > 0 {
                return normalized;
            }
        } except Exception {
        # no-op
        }
        try {
            vals = Object.values(items);
            if vals and vals.length >= 0 {
                return vals;
            }
        } except Exception {
        # no-op
        }
        return normalized;
    }

    def list_len(items: Any) -> int {
        return to_list(items).length;
    }

    def normalize_speed_rows(rows: list) -> list {
        out = [];
        for row in to_list(rows) {
            if not row {
                continue;
            }
            row_name = str(row["name"]) if row else "";
            pytorch_raw = row["pytorch"] if row else None;
            if pytorch_raw is None {
                pytorch_raw = row["baseline"] if row else None;
            }
            pytorch_val = to_num(pytorch_raw);
            if pytorch_val is None {
                continue;
            }

            next_row = {"name": row_name, "pytorch": pytorch_val};

            forged_raw = row["forged"] if row else None;
            if forged_raw is None {
                forged_raw = row["optimized"] if row else None;
            }
            if forged_raw is None {
                forged_raw = row["kernel_ms"] if row else None;
            }
            forged_val = to_num(forged_raw);
            if forged_val is not None {
                next_row["forged"] = forged_val;
            }
            out.push(next_row);
        }
        return out;
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
                charts_req = __jacSpawn(
                    "GetDashboardCharts", "", {"projectName": name}
                );
                status_req = __jacSpawn("GetProjectStatus", "", {"projectName": name});
                hw_req = __jacSpawn("DetectHardware", "", {});
                mcts_req = __jacSpawn(
                    "GetProjectMctsSummary", "", {"projectName": name}
                );
                # Await all responses
                ops_resp = await ops_req;
                charts_resp = await charts_req;
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
                # Process Charts
                charts_data = charts_resp.reports[0]
                if charts_resp and charts_resp.reports
                else {};
                if charts_data {
                    if "operator_usage" in charts_data {
                        ops = to_list(charts_data["operator_usage"]);
                    }
                    if "speed_comparison" in charts_data {
                        setSpeedData(to_list(charts_data["speed_comparison"]));
                    }
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
            }
        } except Exception as e {
            console.error("Failed to fetch dashboard data:", e);
        } finally {
            loading = False;
        }
    }

    useEffect(lambda -> None { refreshData();}, [name]);

    # Auto-refresh status (faster while active, slower while idle)
    useEffect(
        lambda -> Any { poll_ms = get_profile_poll_ms();timer = setInterval(
            lambda -> None { __jacSpawn("GetProjectStatus", "", {"projectName": name}).then(
                lambda resp: any  -> None { data = resp.reports[0]
                if resp and resp.reports
                else {};status = data or {};
                __jacSpawn("GetDashboardCharts", "", {"projectName": name}).then(
                    lambda c_resp: any  -> None { c_data = c_resp.reports[0]
                    if c_resp and c_resp.reports
                    else {};
                    if c_data {
                        if "operator_usage" in c_data {
                            ops = to_list(c_data["operator_usage"]);
                        }
                        if "speed_comparison" in c_data {
                            setSpeedData(to_list(c_data["speed_comparison"]));
                        }
                    }}
                ).catch(
                    lambda e: any  -> None {
                    # console.error("Failed to poll dashboard charts:", e);
                    }
                );
                __jacSpawn("GetProjectMctsSummary", "", {"projectName": name}).then(
                    lambda m_resp: any  -> None { m_data = m_resp.reports[0]
                    if m_resp and m_resp.reports
                    else {};
                    mctsSummary = m_data or {};}
                ).catch(
                    lambda e: any  -> None {
                    # console.error("Failed to poll mcts summary:", e);
                    }
                );}
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

    # Helper for Status Cards
    status_cards = [];

    # MCTS Explorer Card
    status_cards.push(
        <div
            key="mcts"
            className="bg-zinc-900 border border-zinc-800 p-4 rounded-lg cursor-pointer hover:border-indigo-500/50 transition-all"
            onClick={lambda -> None { navigate(f"/project/{name}/operator-workbench");}}
        >
            <div className="flex justify-between items-start mb-2">
                <div className="p-2 bg-indigo-500/10 rounded-lg">
                    <ShieldCheckIcon className="text-indigo-400 w-5 h-5"/>
                </div>
                <span className="text-xs text-zinc-500 bg-zinc-800 px-2 py-1 rounded">
                    MCTS
                </span>
            </div>
            <div className="text-2xl font-bold text-white mb-1">
                {mctsSummary["total_nodes"] if "total_nodes" in mctsSummary else 0}
            </div>
            <div className="text-xs text-zinc-400">
                Nodes Explored
            </div>
        </div>
    );

    # Hardware Card
    status_cards.push(
        <div
            key="hardware"
            className="bg-zinc-900 border border-zinc-800 p-4 rounded-lg flex flex-col justify-between h-full"
        >
            <div>
                <div className="flex justify-between items-start mb-2">
                    <div className="p-2 bg-emerald-500/10 rounded-lg">
                        <CpuIcon className="text-emerald-400 w-5 h-5"/>
                    </div>
                </div>
                <div
                    className="text-lg font-bold text-white mb-2 truncate"
                    title={hardwareStatus}
                >
                    {hardwareStatus}
                </div>
                {<div className="text-xs text-zinc-400 space-y-1 mb-4">
                    <div>
                        Memory:{gpu_mem}
                    </div>
                    <div>
                        Compute:{gpu_compute}
                    </div>
                    <div>
                        SMs:{gpu_sms}
                    </div>
                </div>
                if gpu_mem or gpu_compute or gpu_sms
                else <span></span>}
            </div>
            <div className="mt-2">
                <span className="text-xs text-zinc-500 block mb-1">
                    Backend
                </span>
                <div
                    className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-300"
                >
                    {activeBackend}
                </div>
            </div>
        </div>
    );

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
    bench_pct = bench_progress["percent"] * 100
    if bench_progress and "percent" in bench_progress
    else 0;
    bar_pct = bench_pct;
    if bar_pct > 100 {
        bar_pct = 100;
    }
    if bar_pct < 4 {
        bar_pct = 4;
    }
    bench_bar_width = f"{bar_pct}%";

    gen_job = status["generate"] if status and "generate" in status else {};
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

    jobs = [
        {"key": "profile", "label": "Analysis", "status": profile_norm},
        {
            "key": "generate",
            "label": "Generation",
            "status": generation_step_status
        },
        {
            "key": "optimize",
            "label": "Optimization",
            "status": optimization_step_status
        },
        {"key": "benchmark", "label": "Benchmark", "status": benchmark_step_status}
    ];
    step_items = [];
    step_idx = 0;

    while step_idx < jobs.length {
        job = jobs[step_idx];
        job_status = str(job["status"]).toLowerCase()
        if job and "status" in job and job["status"]
        else "idle";

        is_completed = job_status == "completed";
        is_running = job_status == "running";
        is_failed = job_status == "failed";
        is_active = is_running or is_completed;

        step_icon = <div className="w-2 h-2 rounded-full bg-zinc-600"></div>;
        if is_completed {
            step_icon = <CheckIcon className="w-4 h-4 text-emerald-950"/>;
        } elif is_failed {
            step_icon = <AlertCircleIcon className="w-4 h-4 text-red-950"/>;
        } elif is_running {
            step_icon = <div
                className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse"
            ></div>;
        }

        circle_style = "bg-zinc-800 border-zinc-700 text-zinc-500";
        if is_completed {
            circle_style = "bg-emerald-400 border-emerald-400 text-emerald-950";
        } elif is_failed {
            circle_style = "bg-red-400 border-red-400 text-red-950";
        } elif is_running {
            circle_style = "bg-indigo-500/20 border-indigo-500 text-indigo-400 ring-4 ring-indigo-500/10";
        }

        step_items.push(
            <div
                key={job["key"]}
                className="flex-1 flex flex-col items-center relative"
            >
                <div className="flex items-center w-full">
                    {<div
                        className={"h-0.5 flex-1 " + (
                            "bg-emerald-500/50"
                            if is_completed or is_running or is_failed
                            else "bg-zinc-800"
                        )}
                    ></div>
                    if step_idx > 0
                    else <div className="flex-1"></div>}
                    <div
                        className={"w-8 h-8 rounded-full border-2 flex items-center justify-center z-10 transition-all duration-300 " + circle_style}
                    >
                        {step_icon}
                    </div>
                    {<div
                        className={"h-0.5 flex-1 " + (
                            "bg-emerald-500/50" if is_completed else "bg-zinc-800"
                        )}
                    ></div>
                    if step_idx < jobs.length - 1
                    else <div className="flex-1"></div>}
                </div>
                <div className="mt-3 text-center">
                    <div
                        className={"text-xs font-medium " + (
                            "text-zinc-200"
                            if is_active or is_failed
                            else "text-zinc-500"
                        )}
                    >
                        {job["label"]}
                    </div>
                    <div
                        className={"text-[10px] uppercase tracking-wider font-bold mt-0.5 " + (
                            "text-emerald-400"
                            if is_completed
                            else "text-red-400"
                            if is_failed
                            else "text-indigo-400" if is_running else "text-zinc-600"
                        )}
                    >
                        {job_status if job_status != "idle" else ""}
                    </div>
                </div>
            </div>
        );
        step_idx += 1;
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
        current_message = gen_message if gen_message else "Generating kernels.";
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
        current_message = opt_message if opt_message else "Optimizing kernels.";
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
        current_phase = "completed";
        current_message = "Kernel generation and benchmark completed.";
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
        <div className="p-8 max-w-7xl mx-auto space-y-8">
            <div>
                <h2 className="text-2xl font-bold text-white mb-2">
                    Project Dashboard
                </h2>
                <p className="text-zinc-400">
                    Overview of{name}
                </p>
            </div>
            <div
                className="relative rounded-xl border border-orange-500/30 bg-gradient-to-r from-[#2e1808] via-[#24130a] to-[#171111] p-5 overflow-hidden"
            >
                <div
                    className="absolute -top-16 -right-10 w-56 h-56 rounded-full bg-orange-500/15 blur-3xl"
                ></div>
                <div
                    className="absolute -bottom-20 left-10 w-56 h-56 rounded-full bg-red-500/10 blur-3xl"
                ></div>
                <div className="relative grid grid-cols-1 lg:grid-cols-3 gap-4">
                    <div className="space-y-3">
                        <h3 className="text-sm font-semibold text-orange-200">
                            Kernel Forge Control
                        </h3>
                        <p className="text-xs text-zinc-300 leading-relaxed">
                            Start profiling and baseline benchmarking, then route into operator-level optimization.
                        </p>
                        <div className="flex items-center gap-2">
                            <GlowButton
                                label={(
                                    "Heating forge..."
                                    if forgeBusy
                                    else "Start Forging Kernels"
                                )}
                                onClick={lambda -> None { startForge();}}
                                disabled={forgeBusy
                                or not name
                                or profile_active
                                or gen_active
                                or opt_active}
                            />
                            <button
                                className="px-3 py-2 rounded-lg border border-orange-400/40 bg-orange-500/10 text-orange-200 text-xs font-semibold hover:bg-orange-500/20 transition-colors inline-flex items-center gap-1"
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
                        else <div className="text-xs text-emerald-300">
                            {forgeNotice}
                        </div>
                        if forgeNotice
                        else <span></span>}
                    </div>
                    <div className="space-y-3">
                        <div
                            className="text-xs uppercase tracking-wider font-bold text-zinc-400"
                        >
                            Run Mode
                        </div>
                        <div className="flex items-center gap-2">
                            <button
                                className={"px-3 py-1.5 rounded-md text-xs font-semibold border transition-colors " + (
                                    "border-amber-400 bg-amber-500/20 text-amber-100"
                                    if forgeMode == "auto"
                                    else "border-zinc-700 bg-zinc-900/70 text-zinc-300 hover:bg-zinc-800"
                                )}
                                onClick={lambda -> None { setForgeMode("auto");}}
                            >
                                Automatic
                            </button>
                            <button
                                className={"px-3 py-1.5 rounded-md text-xs font-semibold border transition-colors " + (
                                    "border-amber-400 bg-amber-500/20 text-amber-100"
                                    if forgeMode == "manual"
                                    else "border-zinc-700 bg-zinc-900/70 text-zinc-300 hover:bg-zinc-800"
                                )}
                                onClick={lambda -> None { setForgeMode("manual");}}
                            >
                                Manual
                            </button>
                        </div>
                        {<div className="space-y-2">
                            <div
                                className="text-xs uppercase tracking-wider font-bold text-zinc-400"
                            >
                                Manual Operators
                            </div>
                            <div className="flex items-center gap-2">
                                <button
                                    className="px-2 py-1 rounded-md text-[11px] font-semibold border border-zinc-700 bg-zinc-900/70 text-zinc-300 hover:bg-zinc-800 transition-colors"
                                    onClick={lambda -> None { selectAllManualOps();}}
                                >
                                    Select all
                                </button>
                                <button
                                    className="px-2 py-1 rounded-md text-[11px] font-semibold border border-zinc-700 bg-zinc-900/70 text-zinc-300 hover:bg-zinc-800 transition-colors"
                                    onClick={lambda -> None { clearManualOps();}}
                                >
                                    Clear
                                </button>
                            </div>
                            <div className="max-h-28 overflow-y-auto rounded-md border border-zinc-700 bg-zinc-900/60 p-2 space-y-1">
                                {manualOpItems
                                if list_len(manualOpItems) > 0
                                else <div className="text-xs text-zinc-500">
                                    No operators discovered yet.
                                </div>}
                            </div>
                            <div className="text-[11px] text-zinc-400">
                                Selected: {manualSelectedCount}
                            </div>
                        </div>
                        if forgeMode == "manual"
                        else <div className="space-y-1">
                            <div
                                className="text-xs uppercase tracking-wider font-bold text-zinc-400"
                            >
                                Automatic Scope
                            </div>
                            <div className="text-xs text-zinc-300">
                                Generates and benchmarks all discovered operators.
                            </div>
                        </div>}
                    </div>
                    <div className="space-y-3">
                        <div
                            className="text-xs uppercase tracking-wider font-bold text-zinc-400"
                        >
                            Current State
                        </div>
                        <div
                            className="px-3 py-2 rounded-md border border-zinc-700 bg-zinc-950/80 text-sm text-zinc-100"
                        >
                            {forge_status}
                        </div>
                        <div
                            className={"px-3 py-2 rounded-md border text-xs font-semibold " + (
                                "border-orange-500/50 bg-orange-500/10 text-orange-200"
                                if active_poll
                                else "border-emerald-500/40 bg-emerald-500/10 text-emerald-200"
                                if current_phase == "completed"
                                else "border-zinc-700 bg-zinc-900/70 text-zinc-300"
                            )}
                        >
                            {forge_detail}
                        </div>
                        {<div
                            className="w-full h-1.5 rounded-full bg-zinc-800 overflow-hidden border border-zinc-700"
                        >
                            <div
                                className="h-full bg-gradient-to-r from-orange-400 via-amber-300 to-yellow-300 transition-all duration-500"
                                style={{"width": f"{current_progress_bar}%"}}
                            ></div>
                        </div>
                        if active_poll
                        else <span></span>}
                        {<div className="text-[10px] text-zinc-500 leading-relaxed">
                            {current_state_meta}
                        </div>
                        if current_state_meta
                        else <span></span>}
                        <div className="text-xs text-zinc-400">
                            Full generate/optimize settings live in Operator Workbench modals. Dashboard stays open after profiling unless you open Workbench manually.
                        </div>
                    </div>
                </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {status_cards}
                <div
                    className="col-span-1 md:col-span-2 bg-zinc-900 border border-zinc-800 p-4 rounded-lg"
                >
                    <div className="flex justify-between items-center mb-4">
                        <h3 className="text-sm font-semibold text-zinc-200">
                            Pipeline Status
                        </h3>
                        <div className="text-xs text-zinc-500">
                            Auto-refresh:{poll_label}
                        </div>
                    </div>
                    <div className="flex items-center justify-between px-4 py-6">
                        {step_items}
                    </div>
                </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div
                    className="md:col-span-2 bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden flex flex-col"
                >
                    <div
                        className="p-4 border-b border-zinc-800 flex justify-between items-center bg-zinc-950/50"
                    >
                        <h3 className="font-semibold text-zinc-200">
                            Speed Comparison (ms)
                        </h3>
                        <div className="flex items-center gap-2">
                            <span className="text-xs text-zinc-500">
                                Lower is better
                            </span>
                            <button
                                onClick={lambda -> None { benchLogOpen = True;}}
                                className="text-[11px] px-2 py-1 rounded border border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800 transition-colors"
                            >
                                benchmark.log
                            </button>
                        </div>
                    </div>
                    <div
                        className="p-4 flex-1 flex items-center justify-center min-h-[220px]"
                    >
                        <div className="w-full h-[220px] min-h-[220px]">
                            {<ResponsiveContainer
                                width="100%"
                                height="100%"
                                minWidth={0}
                                minHeight={0}
                            >
                                <BarChart
                                    data={speedData}
                                    margin={{
                                        "top": 5,
                                        "right": 30,
                                        "left": 20,
                                        "bottom": 5
                                    }}
                                >
                                    <CartesianGrid
                                        strokeDasharray="3 3"
                                        stroke="#27272a"
                                        vertical={False}
                                    />
                                    <XAxis
                                        dataKey="name"
                                        stroke="#71717a"
                                        fontSize={12}
                                        tickLine={False}
                                        axisLine={False}
                                        tickFormatter={format_tick_op}
                                    />
                                    <YAxis
                                        stroke="#71717a"
                                        fontSize={12}
                                        tickLine={False}
                                        axisLine={False}
                                        tickFormatter={format_tick_ms}
                                    />
                                    <Tooltip
                                        cursor={{"fill": '#27272a'}}
                                        contentStyle={{
                                            "backgroundColor": '#18181b',
                                            "borderColor": '#27272a',
                                            "borderRadius": '8px',
                                            "color": '#f4f4f5'
                                        }}
                                        labelFormatter={format_tick_op}
                                    />
                                    <Legend/>
                                    <Bar
                                        dataKey="forged"
                                        name="Kernel Forge"
                                        fill="#06b6d4"
                                        stroke="#06b6d4"
                                        fillOpacity={1}
                                        minPointSize={3}
                                        radius={[4, 4, 0, 0]}
                                    />
                                    <Bar
                                        dataKey="pytorch"
                                        name="PyTorch"
                                        fill="#EE4C2C"
                                        minPointSize={3}
                                        radius={[4, 4, 0, 0]}
                                    />
                                </BarChart>
                            </ResponsiveContainer>
                            if list_len(speedData) > 0
                            else <div className="text-zinc-500 text-sm italic">
                                {charts_empty_message}
                            </div>}
                        </div>
                    </div>
                </div>
                <div
                    className="bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden flex flex-col"
                >
                    <div
                        className="p-4 border-b border-zinc-800 flex justify-between items-center bg-zinc-950/50"
                    >
                        <h3 className="font-semibold text-zinc-200">
                            Operator Time Consumption
                        </h3>
                        <span className="text-xs text-zinc-500">
                            Total (ms)
                        </span>
                    </div>
                    <div
                        className="p-4 flex-1 flex items-center justify-center min-h-[220px]"
                    >
                        {<div className="w-full h-[220px] min-h-[220px]">
                            <ResponsiveContainer
                                width="100%"
                                height="100%"
                                minWidth={0}
                                minHeight={0}
                            >
                                <PieChart>
                                    <Pie
                                        data={pieData}
                                        innerRadius={55}
                                        outerRadius={80}
                                        paddingAngle={4}
                                        dataKey="value"
                                    />
                                    <Tooltip
                                        cursor={{"fill": 'rgba(15, 23, 32, 0.6)'}}
                                        contentStyle={{
                                            "backgroundColor": '#0b1218',
                                            "borderColor": '#1f2a36',
                                            "borderRadius": '10px'
                                        }}
                                        itemStyle={{"color": '#e6f2ff'}}
                                        labelStyle={{"color": '#9fb1c7'}}
                                        formatter={format_tick_ms}
                                    />
                                    <Legend
                                        layout="vertical"
                                        verticalAlign="middle"
                                        align="right"
                                        wrapperStyle={{
                                            color: '#dbe7f3',
                                            fontSize: '12px',
                                            paddingLeft: '10px'
                                        }}
                                    />
                                </PieChart>
                            </ResponsiveContainer>
                        </div>
                        if hasPie
                        else <div className="text-zinc-500 text-sm italic">
                            {charts_empty_message}
                        </div>}
                    </div>
                </div>
            </div>
            <Modal
                isOpen={benchLogOpen}
                onClose={lambda -> None { benchLogOpen = False;}}
                title="Benchmark log"
            >
                <div className="p-4 space-y-3">
                    <div className="flex items-center justify-between gap-3">
                        <div
                            className="text-[10px] uppercase tracking-wider font-bold text-zinc-500"
                        >
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
                                className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 rounded-md text-xs text-white border border-indigo-500/40 transition-colors"
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
        </div>;
}
