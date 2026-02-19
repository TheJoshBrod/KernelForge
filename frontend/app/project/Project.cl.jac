"""Project Page Component"""

import from react { useEffect }
import from "react-router-dom" { useNavigate, useParams }
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
import from ...components.backdrops.GameOfLifeBackground { GameOfLifeBackground }
import from ...components.ui.GlowButton { GlowButton }
import from ...components.ui.Icons { CGinSLogo }
import from ...components.ui.Modal { Modal }

def:pub Project -> Any {
    navigate = useNavigate();
    params = useParams();
    raw_name = params.projectName;
    name = raw_name or "";
    COLORS = ['#86e6a9', '#5bbbd3', '#b7f0d0', '#d8f3ff'];

    has ops: list = [],
        loading: bool = True,
        status: dict = {},
        exportPath: str = "",
        searchQuery: str = "",
        downloading: bool = False,
        benchResults: list = [],
        benchLoading: bool = False,
        selectedOps: list = [],
        advOptimize: bool = True,
        advBenchmark: bool = True,
        advIterations: int = 3,
        advPatience: int = 2,
        advMinImprove: float = 0.0,
        advSpeedupPct: float = 5.0,
        advGenAttempts: int = 4,
        advImage: str = "cgins-codex:latest",
        targetDevice: str = "cuda",
        hardwareInfo: dict = {},
        hardwareLoading: bool = False,
        advLog: str = "",
        advLogLoading: bool = False,
        benchLogOpen: bool = False,
        benchLogLoading: bool = False,
        benchLogCopied: bool = False,
        benchLog: str = "";

    def format_op(raw: str) -> str {
        raw_text = raw or "";
        prefix = "torch_nn_functional_";
        if raw_text.startsWith(prefix) {
            return "torch.nn.functional." + raw_text.replace(prefix, "");
        }
        return raw_text.replace("_", ".");
    }

    async def refreshOps -> None {
        loading = True;
        try {
            if name {
                resp = await __jacSpawn("GetProjectOps", "", {"projectName": name});
                data = resp.reports[0]
                if resp and resp.reports and resp.reports.length > 0
                else None;
                if data and data["ops"] {
                    ops = data["ops"];
                } else {
                    ops = [];
                }
            } else {
                ops = [];
            }
        } except Exception as e {
            console.error("Failed to fetch ops:", e);
            ops = [];
        } finally {
            loading = False;
        }
    }

    async def refreshStatus -> None {
        try {
            if name {
                resp = await __jacSpawn("GetProjectStatus", "", {"projectName": name});
                data = resp.reports[0]
                if resp and resp.reports and resp.reports.length > 0
                else {};
                status = data or {};
                adv_job = data["generate_adv"]
                if data and "generate_adv" in data
                else {};
                if adv_job and adv_job["status"] {
                    await refreshAdvLog();
                }
            } else {
                status = {};
            }
        } except Exception as e {
            console.error("Failed to fetch status:", e);
            status = {};
        }
    }

    async def refreshAdvLog -> None {
        if not name {
            return;
        }
        advLogLoading = True;
        try {
            rel_path = f"projects/{name}/logs/generate_adv.log";
            resp = await __jacSpawn(
                "ReadRepoFile", "", {"projectName": name, "relPath": rel_path}
            );
            data = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else {};
            if data and data["success"] {
                content = data["content"] if data and "content" in data else "";
                max_len = 6000;
                if content and content.length > max_len {
                    advLog = content.slice(content.length - max_len);
                } else {
                    advLog = content or "";
                }
            } else {
                advLog = "";
            }
        } except Exception as e {
            console.error("Failed to fetch advanced log:", e);
            advLog = "";
        } finally {
            advLogLoading = False;
        }
    }

    async def refreshBenchLog -> None {
        if not name {
            benchLog = "";
            return;
        }
        benchLogLoading = True;
        try {
            rel_path = f"projects/{name}/logs/benchmark.log";
            resp = await __jacSpawn(
                "ReadRepoFile", "", {"projectName": name, "relPath": rel_path}
            );
            data = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else {};
            if data and "success" in data and data["success"] {
                content = data["content"] if data and "content" in data else "";
                max_len = 8000;
                if content and content.length > max_len {
                    benchLog = content.slice(content.length - max_len);
                } else {
                    benchLog = content or "";
                }
            } else {
                benchLog = "";
            }
        } except Exception as e {
            console.error("Failed to fetch benchmark log:", e);
            benchLog = "";
        } finally {
            benchLogLoading = False;
        }
    }

    async def openBenchLog -> None {
        benchLogOpen = True;
        await refreshBenchLog();
    }

    async def copyBenchLog -> None {
        try {
            await navigator.clipboard.writeText(benchLog or "");
            benchLogCopied = True;
            setTimeout(lambda -> None { benchLogCopied = False;}, 1200);
        } except Exception as e {
            console.error("Failed to copy log:", e);
            alert("Failed to copy log to clipboard.");
        }
    }

    async def refreshBench -> None {
        benchLoading = True;
        try {
            if name {
                resp = await __jacSpawn(
                    "GetProjectBenchmarks", "", {"projectName": name}
                );
                data = resp.reports[0]
                if resp and resp.reports and resp.reports.length > 0
                else {};
                if data and data["results"] {
                    benchResults = data["results"];
                    if selectedOps.length == 0 and data["winners"] {
                        selectedOps = data["winners"];
                    }
                } else {
                    benchResults = [];
                }
            } else {
                benchResults = [];
            }
        } except Exception as e {
            console.error("Failed to fetch benchmarks:", e);
            benchResults = [];
        } finally {
            benchLoading = False;
        }
    }

    async def refreshAll -> None {
        await refreshOps();
        await refreshStatus();
        await refreshBench();
    }

    async def detectHardware -> None {
        hardwareLoading = True;
        try {
            resp = await __jacSpawn("DetectHardware", "", {});
            data = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else {};
            hardwareInfo = data or {};
            if hardwareInfo
            and "mps_available" in hardwareInfo
            and hardwareInfo["mps_available"] {
                targetDevice = "mps";
            } elif hardwareInfo and "cuda_available" in hardwareInfo {
                if not hardwareInfo["cuda_available"] {
                    targetDevice = "cpu";
                } else {
                    targetDevice = "cuda";
                }
            }
        } except Exception as e {
            console.error("Failed to detect hardware:", e);
            hardwareInfo = {};
        } finally {
            hardwareLoading = False;
        }
    }

    useEffect(lambda -> None { refreshAll();detectHardware();}, [name]);
    useEffect(
        lambda -> Any { timer = setInterval(lambda -> None { refreshStatus();}, 4000);return lambda -> None { clearInterval(
            timer
        );}; },
        [name]
    );

    async def handleProfile -> None {
        if name {
            await __jacSpawn("StartProfile", "", {"projectName": name});
            await refreshStatus();
        }
    }

    async def handleGenerate -> None {
        if name {
            await __jacSpawn("StartGenerate", "", {"projectName": name});
            await refreshStatus();
        }
    }

    async def handleGenerateAdvanced -> None {
        if name {
            ops_to_use = [];
            if selectedOps.length > 0 {
                ops_to_use = selectedOps;
            } else {
                for op in ops {
                    if op and op["name"] {
                        ops_to_use = ops_to_use.concat([op["name"]]);
                    }
                }
            }
            advLog = "";
            await __jacSpawn(
                "StartGenerate",
                "",
                {
                    "projectName": name,
                    "ops": ops_to_use,
                    "optimize": advOptimize,
                    "benchmark": advBenchmark,
                    "iterations": advIterations,
                    "patience": advPatience,
                    "minImproveMs": advMinImprove,
                    "minSpeedupPct": advSpeedupPct,
                    "genAttempts": advGenAttempts,
                    "image": advImage,
                    "targetDevice": targetDevice
                }
            );
            await refreshStatus();
        }
    }

    async def handleOptimizeAdvanced -> None {
        if name {
            ops_to_use = [];
            if selectedOps.length > 0 {
                ops_to_use = selectedOps;
            } else {
                for op in ops {
                    if op and op["name"] {
                        ops_to_use = ops_to_use.concat([op["name"]]);
                    }
                }
            }
            await __jacSpawn(
                "StartOptimize",
                "",
                {
                    "projectName": name,
                    "ops": ops_to_use,
                    "iterations": advIterations,
                    "patience": advPatience,
                    "minImproveMs": advMinImprove,
                    "minSpeedupPct": advSpeedupPct,
                    "image": advImage,
                    "targetDevice": targetDevice
                }
            );
            await refreshStatus();
        }
    }

    async def handleOptimize -> None {
        if name {
            await __jacSpawn("StartOptimize", "", {"projectName": name});
            await refreshStatus();
        }
    }

    async def handleBenchmark -> None {
        if name {
            await __jacSpawn("StartBenchmark", "", {"projectName": name});
            await refreshStatus();
            await refreshBench();
        }
    }

    async def setJobControl(jobKey: str, action: str) -> None {
        if name {
            await __jacSpawn(
                "SetJobControl",
                "",
                {"projectName": name, "jobKey": jobKey, "action": action}
            );
            await refreshStatus();
        }
    }

    async def handleExport -> None {
        if name {
            export_ops = selectedOps;
            if export_ops.length == 0 {
                winners = [];
                for r in benchResults {
                    if r["winner"] {
                        winners = winners.concat([r["op"]]);
                    }
                }
                export_ops = winners;
            }
            resp = await __jacSpawn(
                "ExportProject", "", {"projectName": name, "ops": export_ops}
            );
            result = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else None;
            if result and result["success"] {
                exportPath = result["path"];
            }
        }
    }

    async def handleDownloadExport -> None {
        if name {
            downloading = True;
            try {
                export_ops = selectedOps;
                if export_ops.length == 0 {
                    winners = [];
                    for r in benchResults {
                        if r["winner"] {
                            winners = winners.concat([r["op"]]);
                        }
                    }
                    export_ops = winners;
                }
                resp = await __jacSpawn(
                    "DownloadExport", "", {"projectName": name, "ops": export_ops}
                );
                data = resp.reports[0]
                if resp and resp.reports and resp.reports.length > 0
                else None;
                if data and data["success"] {
                    file_name = data["name"] if data["name"] else f"{name}.cgins.zip";
                    data_uri = f"data:application/zip;base64,{data['data']}";
                    blob_resp = await fetch(data_uri);
                    blob = await blob_resp.blob();
                    url = URL.createObjectURL(blob);
                    link = document.createElement("a");
                    link.href = url;
                    link.download = file_name;
                    link.click();
                    URL.revokeObjectURL(url);
                }
            } except Exception as e {
                console.error("Failed to download export:", e);
            } finally {
                downloading = False;
            }
        }
    }

    operatorFreqData = [];
    operatorRows = [];
    hasOperators = False;
    for op in ops {
        operatorFreqData = operatorFreqData.concat(
            [{"operator": format_op(op["name"]), "freq": op["count"]}]
        );
    }

    pieData = [];
    hasPie = False;
    count = 0;
    for op in ops {
        if count < 4 {
            pieData = pieData.concat(
                [
                    {
                        "name": format_op(op["name"]),
                        "value": op["count"],
                        "fill": COLORS[count]
                    }
                ]
            );
            hasPie = True;
            count = count + 1;
        }
    }

    barData = [];
    hasBar = False;

    statusItems = [
        {"key": "profile", "label": "Profile"},
        {"key": "prepare", "label": "Prepare"},
        {"key": "generate", "label": "Generate"},
        {"key": "generate_adv", "label": "Adv Generate"},
        {"key": "optimize_adv", "label": "Adv Optimize"},
        {"key": "optimize", "label": "Optimize"},
        {"key": "benchmark", "label": "Benchmark"}
    ];

    def status_label(key: str) -> str {
        job = status[key];
        if job and job["status"] {
            return job["status"];
        }
        return "idle";
    }

    statusRows = [];
    for item in statusItems {
        job = status[item["key"]];
        progress = job["progress"] if job and "progress" in job else {};
        total = progress["total"] if progress and "total" in progress else 0;
        current = progress["current"] if progress and "current" in progress else 0;
        pct = progress["percent"] * 100 if progress and "percent" in progress else 0;
        bar_width = f"{pct}%" if pct > 2 else "8%";
        show_progress = (
            (
                item["key"] == "generate"
                or item["key"] == "generate_adv"
                or item["key"] == "optimize_adv"
                or item["key"] == "optimize"
                or item["key"] == "benchmark"
            )
            and job
            and job["status"] == "running"
        );
        message = job["message"] if job and "message" in job else "";
        progress_text = f"{current}/{total}" if total > 0 else "working...";
        controls = <span></span>;
        if (
            item["key"] == "generate"
            or item["key"] == "generate_adv"
            or item["key"] == "optimize_adv"
            or item["key"] == "optimize"
        ) {
            if job and job["status"] == "running" {
                controls = <div className="status-controls">
                    <button
                        onClick={lambda -> None { setJobControl(item["key"], "pause");}}
                        className="chip-button"
                    >
                        Pause
                    </button>
                    <button
                        onClick={lambda -> None { setJobControl(item["key"], "cancel");}}
                        className="chip-button chip-danger"
                    >
                        Cancel
                    </button>
                </div>;
            } elif job and job["status"] == "paused" {
                controls = <div className="status-controls">
                    <button
                        onClick={lambda -> None { setJobControl(item["key"], "resume");}}
                        className="chip-button"
                    >
                        Resume
                    </button>
                    <button
                        onClick={lambda -> None { setJobControl(item["key"], "cancel");}}
                        className="chip-button chip-danger"
                    >
                        Cancel
                    </button>
                </div>;
            }
        }
        statusRows = statusRows.concat(
            [
                <div key={item["key"]} className="status-row">
                    <div className="status-row-header">
                        <span className="status-label">
                            {item["label"]}
                        </span>
                        <span className="status-pill">
                            {status_label(item["key"])}
                        </span>
                    </div>
                    {<div className="status-progress">
                        <div className="progress-track">
                            <div
                                className="progress-fill"
                                style={{"width": bar_width}}
                            ></div>
                        </div>
                        <div className="progress-meta">
                            <span>
                                {progress_text}
                            </span>
                            <span className="progress-message">
                                {message}
                            </span>
                        </div>
                        {controls}
                    </div>
                    if show_progress or (job and job["status"] == "paused")
                    else <span></span>}
                </div>
            ]
        );
    }

    def _job_time(job: dict) -> float {
        if job
        and "progress" in job
        and job["progress"]
        and "updated_at" in job["progress"] {
            return Number(job["progress"]["updated_at"]);
        }
        if job and "finished_at" in job {
            return Number(job["finished_at"]);
        }
        if job and "started_at" in job {
            return Number(job["started_at"]);
        }
        return 0.0;
    }

    def _usage_available(usage: dict) -> bool {
        return usage
        and (
            "input_tokens" in usage
            or "output_tokens" in usage
            or "reasoning_tokens" in usage
            or "cached_tokens" in usage
        );
    }

    def _usage_value(usage: dict, key: str) -> int {
        if usage and key in usage {
            return usage[key];
        }
        return 0;
    }

    def _usage_label(key: str) -> str {
        if key == "optimize_adv" {
            return "Adv Optimize";
        }
        if key == "generate_adv" {
            return "Adv Generate";
        }
        return key;
    }

    usageData = {};
    usageJob = "";
    usageTime = 0.0;
    for key in ["optimize_adv", "generate_adv"] {
        job = status[key] if status and key in status else {};
        usage = job["usage"] if job and "usage" in job else {};
        if _usage_available(usage) {
            t = _job_time(job);
            if t >= usageTime {
                usageTime = t;
                usageData = usage;
                usageJob = key;
            }
        }
    }

    bench_job = status["benchmark"];
    bench_progress = bench_job["progress"]
    if bench_job and "progress" in bench_job
    else {};
    bench_total = bench_progress["total"]
    if bench_progress and "total" in bench_progress
    else 0;
    bench_current = bench_progress["current"]
    if bench_progress and "current" in bench_progress
    else 0;
    bench_pct = bench_progress["percent"] * 100
    if bench_progress and "percent" in bench_progress
    else 0;
    # Keep a small visible fill while progress is initializing, without shrinking.
    bar_pct = bench_pct;
    if bar_pct > 100 {
        bar_pct = 100;
    }
    if bar_pct < 4 {
        bar_pct = 4;
    }
    bench_bar_width = f"{bar_pct}%";
    bench_message = bench_job["message"]
    if bench_job and "message" in bench_job
    else "";
    bench_progress_text = f"{bench_current}/{bench_total}"
    if bench_total > 0
    else "working...";
    show_bench_progress = (
        bench_job
        and (bench_job["status"] == "running" or bench_job["status"] == "paused")
    );

    opLinks = [];
    hasOps = False;
    filtered_ops = [];
    for op in ops {
        if not searchQuery or format_op(op["name"]).includes(searchQuery) {
            filtered_ops = filtered_ops.concat([op]);
        }
    }
    for op in filtered_ops {
        op_name = op["name"];
        path = f"/project/{name}/{op_name}";
        opLinks = opLinks.concat(
            [
                <a
                    key={op["name"]}
                    onClick={lambda -> None { navigate(path);}}
                    className="op-item"
                >
                    {format_op(op["name"])}
                </a>
            ]
        );
        hasOps = True;
    }

    opSelectItems = [];
    hasOpSelect = False;
    for op in filtered_ops {
        op_name = op["name"];
        checked = selectedOps.includes(op_name);
        opSelectItems = opSelectItems.concat(
            [
                <label key={op_name} className="checkbox-row">
                    <input
                        type="checkbox"
                        checked={checked}
                        onChange={lambda e: any  -> None { toggleOpSelection(op_name);}}
                    />
                    <span className="mono">
                        {format_op(op_name)}
                    </span>
                </label>
            ]
        );
        hasOpSelect = True;
    }

    for row in operatorFreqData {
        operatorRows = operatorRows.concat(
            [
                <tr key={row["operator"]} className="table-row">
                    <td className="table-cell mono">
                        {row["operator"]}
                    </td>
                    <td className="table-cell">
                        {row["freq"]}
                    </td>
                </tr>
            ]
        );
        hasOperators = True;
    }

    def toggleOpSelection(op_name: str) -> None {
        if selectedOps.includes(op_name) {
            new_list = [];
            for o in selectedOps {
                if o != op_name {
                    new_list = new_list.concat([o]);
                }
            }
            selectedOps = new_list;
        } else {
            new_list = [];
            for o in selectedOps {
                new_list = new_list.concat([o]);
            }
            new_list = new_list.concat([op_name]);
            selectedOps = new_list;
        }
    }

    def selectAllOps -> None {
        all_ops = [];
        for op in ops {
            if op and op["name"] {
                all_ops = all_ops.concat([op["name"]]);
            }
        }
        selectedOps = all_ops;
    }

    def clearSelectedOps -> None {
        selectedOps = [];
    }

    benchRows = [];
    hasBench = False;
    for r in benchResults {
        op_name = r["op"];
        speedup = r["speedup"];
        kernel_status = r["kernel_status"] if r and "kernel_status" in r else "";
        kernel_error = r["kernel_error"] if r and "kernel_error" in r else "";
        kernel_ms = r["kernel_ms"];
        kernel_text = "n/a";
        kernel_class = "table-muted";
        if kernel_status == "ok" {
            kernel_text = f"{kernel_ms:.6f}" if kernel_ms else "0";
            kernel_class = "";
        } elif kernel_status == "missing" {
            kernel_text = "missing";
            kernel_class = "table-muted";
        } elif kernel_status == "error" {
            kernel_text = "error";
            kernel_class = "table-error";
        } elif kernel_status == "skipped" {
            kernel_text = "skipped";
            kernel_class = "table-muted";
        }
        speedup_text = (
            f"{speedup:.2f}x" if (kernel_status == "ok" and speedup) else "n/a"
        );
        is_win = r["winner"];
        source_text = r["source"] if r and r["source"] else "";
        if kernel_status == "missing" {
            source_text = "missing";
        } elif kernel_status == "error" {
            source_text = f"{source_text} (error)";
        }
        checked = selectedOps.includes(op_name);
        benchRows = benchRows.concat(
            [
                <tr key={op_name} className="table-row">
                    <td className="table-cell mono">
                        {format_op(op_name)}
                    </td>
                    <td className="table-cell">
                        {r["pytorch_ms"]}
                    </td>
                    <td className={"table-cell " + kernel_class}>
                        <span title={kernel_error}>
                            {kernel_text}
                        </span>
                    </td>
                    <td className="table-cell">
                        <span
                            className={("status-success" if is_win else "table-muted")}
                        >
                            {speedup_text}
                        </span>
                    </td>
                    <td className="table-cell table-muted">
                        {source_text}
                    </td>
                    <td className="table-cell text-center">
                        <input
                            type="checkbox"
                            checked={checked}
                            onChange={lambda e: any  -> None { toggleOpSelection(
                                op_name
                            );}}
                        />
                    </td>
                </tr>
            ]
        );
        hasBench = True;
    }

    hardwareStatus = "";
    if hardwareInfo
    and "mps_available" in hardwareInfo
    and hardwareInfo["mps_available"] {
        hardwareStatus = "MPS: Apple Silicon";
    } elif hardwareInfo and "cuda_available" in hardwareInfo {
        if hardwareInfo["cuda_available"] {
            gpu_name = hardwareInfo["device_name"]
            if "device_name" in hardwareInfo
            else "";
            count = hardwareInfo["device_count"]
            if "device_count" in hardwareInfo
            else 0;
            suffix = f" x{count}" if count and count > 1 else "";
            hardwareStatus = f"CUDA: {gpu_name}{suffix}";
        } else {
            hardwareStatus = "CUDA not detected";
        }
    }

    skeletonOps = [];
    for i in [0, 1, 2, 3, 4, 5] {
        skeletonOps = skeletonOps.concat(
            [<div key={i} className="skeleton skeleton-line"></div>]
        );
    }

    return
        <div className="app-shell">
            <GameOfLifeBackground/>
            <div className="app-backdrop"></div>
            <div className="app-layout app-content-layer">
                <aside className="sidebar">
                    <div className="sidebar-header">
                        <div className="brand">
                            <CGinSLogo className="mascot mascot--logo"/>
                            <div className="brand-meta">
                                <span>
                                    Project
                                </span>
                                <span className="brand-tag">
                                    {name}
                                </span>
                            </div>
                        </div>
                        <div className="sidebar-actions">
                            <button
                                onClick={lambda -> None { navigate("/");}}
                                className="ghost-button"
                            >
                                Home
                            </button>
                            <button
                                onClick={lambda -> None { navigate("/settings");}}
                                className="ghost-button"
                            >
                                Settings
                            </button>
                        </div>
                    </div>
                    <div className="panel sidebar-panel">
                        <div className="panel-header">
                            <span className="panel-title">
                                Operators
                            </span>
                            <button
                                onClick={lambda -> None { refreshOps();}}
                                className="chip-button"
                            >
                                Refresh
                            </button>
                        </div>
                        <div className="panel-body">
                            <input
                                type="text"
                                value={searchQuery}
                                onChange={lambda e: any  -> None { searchQuery = e.target.value;}}
                                placeholder="Search ops..."
                                className="input"
                            />
                            <div className="op-list">
                                {skeletonOps
                                if loading
                                else <p className="table-muted">
                                    No ops profiled yet. Run the profiling step.
                                </p>
                                if not hasOps
                                else opLinks}
                            </div>
                        </div>
                    </div>
                    <div className="panel sidebar-panel">
                        <div className="panel-header">
                            <span className="panel-title">
                                Advanced Selection
                            </span>
                            <span className="table-muted">
                                {selectedOps.length}selected
                            </span>
                        </div>
                        <div className="panel-body">
                            <div className="button-row">
                                <button
                                    onClick={lambda -> None { selectAllOps();}}
                                    className="chip-button"
                                >
                                    Select All
                                </button>
                                <button
                                    onClick={lambda -> None { clearSelectedOps();}}
                                    className="chip-button"
                                >
                                    Clear
                                </button>
                            </div>
                            <div className="checkbox-list">
                                {<p className="table-muted">
                                    No ops to select yet.
                                </p>
                                if not hasOpSelect
                                else opSelectItems}
                            </div>
                        </div>
                    </div>
                    <div className="panel sidebar-panel">
                        <div className="panel-header">
                            <span className="panel-title">
                                Advanced Generate
                            </span>
                            <span className="table-muted">
                                Codex loop
                            </span>
                        </div>
                        <div className="panel-body">
                            <label className="checkbox-row">
                                <input
                                    type="checkbox"
                                    checked={advBenchmark}
                                    onChange={lambda e: any  -> None { advBenchmark = not advBenchmark;}}
                                />
                                <span>
                                    Benchmark after generate
                                </span>
                            </label>
                            <label className="checkbox-row">
                                <input
                                    type="checkbox"
                                    checked={advOptimize}
                                    onChange={lambda e: any  -> None { advOptimize = not advOptimize;}}
                                />
                                <span>
                                    Optimize with Codex loop
                                </span>
                            </label>
                            <label className="input-label">
                                Optimize iterations
                            </label>
                            <input
                                type="number"
                                min="1"
                                value={advIterations}
                                onChange={lambda e: any  -> None { advIterations = Number(
                                    e.target.value
                                );}}
                                className="input"
                            />
                            <label className="input-label">
                                No-improve patience
                            </label>
                            <input
                                type="number"
                                min="1"
                                value={advPatience}
                                onChange={lambda e: any  -> None { advPatience = Number(
                                    e.target.value
                                );}}
                                className="input"
                            />
                            <label className="input-label">
                                Min improve (ms)
                            </label>
                            <input
                                type="number"
                                min="0"
                                step="0.001"
                                value={advMinImprove}
                                onChange={lambda e: any  -> None { advMinImprove = Number(
                                    e.target.value
                                );}}
                                className="input"
                            />
                            <label className="input-label">
                                Target speedup (%)
                            </label>
                            <input
                                type="number"
                                min="0"
                                step="0.1"
                                value={advSpeedupPct}
                                onChange={lambda e: any  -> None { advSpeedupPct = Number(
                                    e.target.value
                                );}}
                                className="input"
                            />
                            <label className="input-label">
                                Generate attempts
                            </label>
                            <input
                                type="number"
                                min="1"
                                value={advGenAttempts}
                                onChange={lambda e: any  -> None { advGenAttempts = Number(
                                    e.target.value
                                );}}
                                className="input"
                            />
                            <label className="input-label">
                                Target device
                            </label>
                            {<label className="checkbox-row">
                                <input
                                    type="radio"
                                    checked={targetDevice == "mps"}
                                    onChange={lambda e: any  -> None { targetDevice = "mps";}}
                                />
                                <span>
                                    Apple MPS
                                </span>
                            </label>
                            if hardwareInfo
                            and "mps_available" in hardwareInfo
                            and hardwareInfo["mps_available"]
                            else <span></span>}
                            <label className="checkbox-row">
                                <input
                                    type="radio"
                                    checked={targetDevice == "cuda"}
                                    onChange={lambda e: any  -> None { targetDevice = "cuda";}}
                                />
                                <span>
                                    GPU (CUDA)
                                </span>
                            </label>
                            <label className="checkbox-row">
                                <input
                                    type="radio"
                                    checked={targetDevice == "cpu"}
                                    onChange={lambda e: any  -> None { targetDevice = "cpu";}}
                                />
                                <span>
                                    CPU
                                </span>
                            </label>
                            <button
                                onClick={lambda -> None { detectHardware();}}
                                className="ghost-button"
                            >
                                {"Detecting..."
                                if hardwareLoading
                                else "Detect Hardware"}
                            </button>
                            {<p className="table-muted">
                                {hardwareStatus}
                            </p>
                            if hardwareStatus
                            else <span></span>}
                            <label className="input-label">
                                Docker image
                            </label>
                            <input
                                type="text"
                                value={advImage}
                                onChange={lambda e: any  -> None { advImage = e.target.value;}}
                                className="input"
                            />
                            <GlowButton
                                label="Advanced Generate"
                                onClick={lambda -> None { handleGenerateAdvanced();}}
                                variant="primary"
                            />
                            <GlowButton
                                label="Advanced Optimize"
                                onClick={lambda -> None { handleOptimizeAdvanced();}}
                                variant="secondary"
                            />
                        </div>
                    </div>
                    <div className="panel sidebar-panel">
                        <div className="panel-header">
                            <span className="panel-title">
                                Pipeline
                            </span>
                            <span className="table-muted">
                                Actions
                            </span>
                        </div>
                        <div className="panel-body button-stack">
                            <GlowButton
                                label="Profile Model"
                                onClick={lambda -> None { handleProfile();}}
                                variant="primary"
                            />
                            <button
                                onClick={lambda -> None { handleGenerate();}}
                                className="ghost-button"
                            >
                                Generate Kernels
                            </button>
                            <button
                                onClick={lambda -> None { handleOptimize();}}
                                className="ghost-button"
                            >
                                Optimize Kernels
                            </button>
                            <button
                                onClick={lambda -> None { handleBenchmark();}}
                                className="ghost-button"
                            >
                                Benchmark
                            </button>
                            <button
                                onClick={lambda -> None { handleExport();}}
                                className="ghost-button"
                            >
                                Export .cgins
                            </button>
                            {<button
                                onClick={lambda -> None { handleDownloadExport();}}
                                disabled={downloading}
                                className="ghost-button"
                            >
                                {"Downloading..." if downloading else "Download Export"}
                            </button>
                            if exportPath
                            else <span></span>}
                            {<p className="table-muted break-all">
                                {exportPath}
                            </p>
                            if exportPath
                            else <span></span>}
                        </div>
                    </div>
                </aside>
                <main className="content">
                    <div className="page-header">
                        <div>
                            <h1 className="page-title">
                                Project:{name}
                            </h1>
                            <div className="badge-row">
                                <span className="badge">
                                    Ops:{ops.length}
                                </span>
                                <span className="badge">
                                    Benchmarks:{benchResults.length}
                                </span>
                            </div>
                        </div>
                        <CGinSLogo className="mascot mascot--cta"/>
                    </div>
                    <div className="panel-grid">
                        <div className="panel">
                            <div className="panel-header">
                                <span className="panel-title">
                                    Operator usage
                                </span>
                                <span className="table-muted">
                                    Top 4 ops
                                </span>
                            </div>
                            <div className="panel-body">
                                {<div className="skeleton skeleton-block"></div>
                                if loading
                                else <p className="table-muted">
                                    No profiling data yet.
                                </p>
                                if not hasPie
                                else <div className="chart-wrap">
                                    <ResponsiveContainer width="100%" height={220}>
                                        <PieChart>
                                            <Pie
                                                data={pieData}
                                                innerRadius={55}
                                                outerRadius={80}
                                                paddingAngle={4}
                                                dataKey="value"
                                            />
                                            <Tooltip
                                                cursor={{
                                                    "fill": 'rgba(15, 23, 32, 0.6)'
                                                }}
                                                contentStyle={{
                                                    "backgroundColor": '#0b1218',
                                                    "borderColor": '#1f2a36',
                                                    "borderRadius": '10px'
                                                }}
                                                itemStyle={{"color": '#e6f2ff'}}
                                                labelStyle={{"color": '#9fb1c7'}}
                                            />
                                            <Legend
                                                layout="vertical"
                                                verticalAlign="middle"
                                                align="right"
                                                wrapperStyle={{"color": '#dbe7f3'}}
                                            />
                                        </PieChart>
                                    </ResponsiveContainer>
                                </div>}
                            </div>
                        </div>
                        <div className="panel">
                            <div className="panel-header">
                                <span className="panel-title">
                                    Pipeline status
                                </span>
                                <button
                                    onClick={lambda -> None { refreshStatus();}}
                                    className="chip-button"
                                >
                                    Refresh
                                </button>
                            </div>
                            <div className="panel-body status-list">
                                {<div className="usage-summary">
                                    <div className="usage-header">
                                        <span className="usage-title">
                                            Latest token usage
                                        </span>
                                        <span className="usage-source">
                                            {_usage_label(usageJob)}
                                        </span>
                                    </div>
                                    <div className="usage-grid">
                                        <div className="usage-card">
                                            <span className="usage-label">
                                                Input
                                            </span>
                                            <span className="usage-value">
                                                {_usage_value(
                                                    usageData, "input_tokens"
                                                )}
                                            </span>
                                        </div>
                                        <div className="usage-card">
                                            <span className="usage-label">
                                                Output
                                            </span>
                                            <span className="usage-value">
                                                {_usage_value(
                                                    usageData, "output_tokens"
                                                )}
                                            </span>
                                        </div>
                                        <div className="usage-card">
                                            <span className="usage-label">
                                                Reasoning
                                            </span>
                                            <span className="usage-value">
                                                {_usage_value(
                                                    usageData, "reasoning_tokens"
                                                )}
                                            </span>
                                        </div>
                                        <div className="usage-card">
                                            <span className="usage-label">
                                                Cached
                                            </span>
                                            <span className="usage-value">
                                                {_usage_value(
                                                    usageData, "cached_tokens"
                                                )}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                                if usageJob
                                else <span></span>}{statusRows}
                            </div>
                        </div>
                    </div>
                    <div className="panel">
                        <div className="panel-header">
                            <span className="panel-title">
                                Operator frequency
                            </span>
                            <span className="table-muted">
                                All operators
                            </span>
                        </div>
                        <div className="panel-body">
                            <table className="table">
                                <thead className="table-head">
                                    <tr>
                                        <th className="table-cell">
                                            Operator
                                        </th>
                                        <th className="table-cell">
                                            Freq
                                        </th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {operatorRows}
                                </tbody>
                            </table>
                        </div>
                    </div>
                    <div className="panel">
                        <div className="panel-header">
                            <span className="panel-title">
                                Benchmark results
                            </span>
                            <div className="button-row">
                                <button
                                    onClick={lambda -> None { openBenchLog();}}
                                    className="chip-button"
                                >
                                    View log
                                </button>
                                <button
                                    onClick={lambda -> None { refreshBench();}}
                                    className="chip-button"
                                >
                                    Refresh
                                </button>
                            </div>
                        </div>
                        <div className="panel-body">
                            {<div className="skeleton skeleton-block"></div>
                            if benchLoading
                            else <p className="table-muted">
                                No benchmark data yet. Run the Benchmark step.
                            </p>
                            if not hasBench
                            else <table className="table">
                                <thead className="table-head">
                                    <tr>
                                        <th className="table-cell">
                                            Op
                                        </th>
                                        <th className="table-cell">
                                            PyTorch (ms)
                                        </th>
                                        <th className="table-cell">
                                            Kernel (ms)
                                        </th>
                                        <th className="table-cell">
                                            Speedup
                                        </th>
                                        <th className="table-cell">
                                            Source
                                        </th>
                                        <th className="table-cell text-center">
                                            Export
                                        </th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {benchRows}
                                </tbody>
                            </table>}{<div className="status-progress">
                                <div className="progress-track">
                                    <div
                                        className="progress-fill"
                                        style={{"width": bench_bar_width}}
                                    ></div>
                                </div>
                                <div className="progress-meta">
                                    <span>
                                        {bench_progress_text}
                                    </span>
                                    <span className="progress-message">
                                        {bench_message}
                                    </span>
                                </div>
                            </div>
                            if show_bench_progress
                            else <span></span>}
                        </div>
                        <div className="panel-footer">
                            <p className="table-muted">
                                Default export includes ops where kernel beats PyTorch.
                            </p>
                            <p className="table-muted">
                                Kernel errors are shown as "error" (see benchmark log for details).
                            </p>
                        </div>
                    </div>
                    <div className="panel-grid">
                        <div className="panel">
                            <div className="panel-header">
                                <span className="panel-title">
                                    Advanced generate log
                                </span>
                                <button
                                    onClick={lambda -> None { refreshAdvLog();}}
                                    className="chip-button"
                                >
                                    Refresh
                                </button>
                            </div>
                            <div className="panel-body">
                                {<div className="skeleton skeleton-block"></div>
                                if advLogLoading
                                else <pre className="code-block">
                                    {advLog or "No log yet."}
                                </pre>}
                            </div>
                        </div>
                        <div className="panel">
                            <div className="panel-header">
                                <span className="panel-title">
                                    Time vs. version
                                </span>
                                <span className="table-muted">
                                    Optimization
                                </span>
                            </div>
                            <div className="panel-body">
                                {<p className="table-muted">
                                    No optimization data yet.
                                </p>
                                if not hasBar
                                else <div className="chart-wrap">
                                    <ResponsiveContainer width="100%" height={220}>
                                        <BarChart data={barData}>
                                            <CartesianGrid
                                                strokeDasharray="3 3"
                                                stroke="rgba(148, 163, 184, 0.18)"
                                                vertical={False}
                                            />
                                            <XAxis
                                                dataKey="name"
                                                stroke="#8fa1b5"
                                                tick={{
                                                    "fill": '#8fa1b5',
                                                    "fontSize": 12
                                                }}
                                                axisLine={False}
                                                tickLine={False}
                                            />
                                            <YAxis
                                                stroke="#8fa1b5"
                                                tick={{
                                                    "fill": '#8fa1b5',
                                                    "fontSize": 12
                                                }}
                                                axisLine={False}
                                                tickLine={False}
                                            />
                                            <Tooltip
                                                cursor={{
                                                    "fill": 'rgba(15, 23, 32, 0.6)'
                                                }}
                                                contentStyle={{
                                                    "backgroundColor": '#0b1218',
                                                    "borderColor": '#1f2a36',
                                                    "borderRadius": '10px'
                                                }}
                                                itemStyle={{"color": '#e6f2ff'}}
                                                labelStyle={{"color": '#9fb1c7'}}
                                            />
                                            <Bar
                                                dataKey="time"
                                                radius={[6, 6, 0, 0]}
                                                barSize={32}
                                                fill="#86e6a9"
                                            />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>}
                            </div>
                        </div>
                    </div>
                </main>
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
