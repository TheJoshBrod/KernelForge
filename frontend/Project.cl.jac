"""Project Page Component"""

import from react { useEffect }
import from "@jac/runtime" { jacSpawn }
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

def:pub Project -> any {
    navigate = useNavigate();
    params = useParams();
    raw_name = params.projectName;
    name = raw_name or "";
    COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

    has ops: list = [],
        loading: bool = True,
        status: dict = {},
        exportPath: str = "",
        searchQuery: str = "",
        downloading: bool = False,
        benchResults: list = [],
        benchLoading: bool = False,
        selectedOps: list = [];

    def format_op(raw: str) -> str {
        raw_text = raw or "";
        prefix = "torch_nn_functional_";
        if raw_text.startswith(prefix) {
            return "torch.nn.functional." + raw_text.replace(prefix, "");
        }
        return raw_text.replace("_", ".");
    }

    async def refreshOps -> None {
        loading = True;
        try {
            if name {
                resp = await jacSpawn("GetProjectOps", "", {"projectName": name});
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
        } except (error) {
            console.error("Failed to fetch ops:", error);
            ops = [];
        } finally {
            loading = False;
        }
    }

    async def refreshStatus -> None {
        try {
            if name {
                resp = await jacSpawn("GetProjectStatus", "", {"projectName": name});
                data = resp.reports[0]
                if resp and resp.reports and resp.reports.length > 0
                else {};
                status = data or {};
            } else {
                status = {};
            }
        } except (error) {
            console.error("Failed to fetch status:", error);
            status = {};
        }
    }

    async def refreshBench -> None {
        benchLoading = True;
        try {
            if name {
                resp = await jacSpawn(
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
        } except (error) {
            console.error("Failed to fetch benchmarks:", error);
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

    useEffect(lambda -> None { refreshAll();}, [name]);
    useEffect(
        lambda -> None { timer = setInterval(lambda -> None { refreshStatus();}, 4000);return lambda -> None { clearInterval(
            timer
        );}; },
        [name]
    );

    async def handleProfile -> None {
        if name {
            await jacSpawn("StartProfile", "", {"projectName": name});
            await refreshStatus();
        }
    }

    async def handleGenerate -> None {
        if name {
            await jacSpawn("StartGenerate", "", {"projectName": name});
            await refreshStatus();
        }
    }

    async def handleOptimize -> None {
        if name {
            await jacSpawn("StartOptimize", "", {"projectName": name});
            await refreshStatus();
        }
    }

    async def handleBenchmark -> None {
        if name {
            await jacSpawn("StartBenchmark", "", {"projectName": name});
            await refreshStatus();
            await refreshBench();
        }
    }

    async def setJobControl(jobKey: str, action: str) -> None {
        if name {
            await jacSpawn(
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
            resp = await jacSpawn(
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
                resp = await jacSpawn(
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
            } except (error) {
                console.error("Failed to download export:", error);
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
                        "fill": COLORS[0]
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
        {"key": "generate", "label": "Generate"},
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
                or item["key"] == "optimize"
                or item["key"] == "benchmark"
            )
            and job
            and job["status"] == "running"
        );
        message = job["message"] if job and "message" in job else "";
        progress_text = f"{current}/{total}" if total > 0 else "working...";
        controls = <span></span>;
        if item["key"] == "generate" or item["key"] == "optimize" {
            if job and job["status"] == "running" {
                controls = <div className="flex gap-2 text-xs">
                    <button
                        onClick={lambda -> None { setJobControl(item["key"], "pause");}}
                        className="px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700"
                    >
                        Pause
                    </button>
                    <button
                        onClick={lambda -> None { setJobControl(item["key"], "cancel");}}
                        className="px-2 py-1 rounded bg-red-900/60 hover:bg-red-900"
                    >
                        Cancel
                    </button>
                </div>;
            } elif job and job["status"] == "paused" {
                controls = <div className="flex gap-2 text-xs">
                    <button
                        onClick={lambda -> None { setJobControl(item["key"], "resume");}}
                        className="px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700"
                    >
                        Resume
                    </button>
                    <button
                        onClick={lambda -> None { setJobControl(item["key"], "cancel");}}
                        className="px-2 py-1 rounded bg-red-900/60 hover:bg-red-900"
                    >
                        Cancel
                    </button>
                </div>;
            }
        }
        statusRows = statusRows.concat(
            [
                <div key={item["key"]} className="text-sm text-zinc-300 space-y-1">
                    <div className="flex items-center justify-between">
                        <span>
                            {item["label"]}
                        </span>
                        <span className="font-mono text-xs text-zinc-500">
                            {status_label(item["key"])}
                        </span>
                    </div>
                    {<div className="space-y-1">
                        <div
                            className="h-2 w-full rounded bg-zinc-800 overflow-hidden"
                        >
                            <div
                                className="h-2 bg-emerald-500"
                                style={{"width": bar_width}}
                            ></div>
                        </div>
                        <div
                            className="flex justify-between text-[10px] text-zinc-500"
                        >
                            <span>
                                {progress_text}
                            </span>
                            <span className="truncate max-w-[140px]">
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
    bench_bar_width = f"{bench_pct}%" if bench_pct > 2 else "8%";
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
                    className="block w-full text-left bg-zinc-900/50 hover:bg-zinc-800 border border-zinc-800/50 rounded px-4 py-3 text-sm font-mono text-zinc-300 transition-all cursor-pointer"
                >
                    {format_op(op["name"])}
                </a>
            ]
        );
        hasOps = True;
    }

    for row in operatorFreqData {
        operatorRows = operatorRows.concat(
            [
                <tr
                    key={row["operator"]}
                    className="hover:bg-zinc-800/50 transition-colors"
                >
                    <td className="px-6 py-3 font-mono text-zinc-300">
                        {row["operator"]}
                    </td>
                    <td className="px-6 py-3 text-zinc-300">
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
            selectedOps = selectedOps.concat([op_name]);
        }
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
        kernel_class = "text-zinc-400";
        if kernel_status == "ok" {
            kernel_text = f"{kernel_ms:.6f}" if kernel_ms else "0";
            kernel_class = "text-zinc-300";
        } elif kernel_status == "missing" {
            kernel_text = "missing";
            kernel_class = "text-zinc-500";
        } elif kernel_status == "error" {
            kernel_text = "error";
            kernel_class = "text-red-400";
        } elif kernel_status == "skipped" {
            kernel_text = "skipped";
            kernel_class = "text-zinc-500";
        }
        speedup_text = (
            f"{speedup:.2f}x" if (kernel_status == "ok" and speedup) else "n/a"
        );
        is_win = r["winner"];
        source_text = r["source"] if r and r["source"] else "";
        if kernel_status == "missing" {
            source_text = "missing";
        } elif kernel_status == "error" {
            source_text = source_text + " (error)";
        }
        checked = selectedOps.includes(op_name);
        benchRows = benchRows.concat(
            [
                <tr key={op_name} className="hover:bg-zinc-800/50 transition-colors">
                    <td className="px-4 py-2 font-mono text-xs text-zinc-200">
                        {format_op(op_name)}
                    </td>
                    <td className="px-4 py-2 text-xs text-zinc-400">
                        {r["pytorch_ms"]}
                    </td>
                    <td className={"px-4 py-2 text-xs " + kernel_class}>
                        <span title={kernel_error}>
                            {kernel_text}
                        </span>
                    </td>
                    <td className="px-4 py-2 text-xs">
                        <span
                            className={(
                                "text-emerald-400" if is_win else "text-zinc-500"
                            )}
                        >
                            {speedup_text}
                        </span>
                    </td>
                    <td className="px-4 py-2 text-xs text-zinc-500">
                        {source_text}
                    </td>
                    <td className="px-4 py-2 text-xs text-center">
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

    return
        <div className="flex min-h-screen bg-zinc-950 text-zinc-100 font-sans">
            <div className="absolute top-2 right-6">
                <a
                    onClick={lambda -> None { navigate("/settings");}}
                    className="flex items-center justify-center p-2 rounded-full hover:bg-zinc-800 transition-colors text-zinc-400 hover:text-white cursor-pointer"
                    aria-label="Settings"
                >
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    >
                        <path
                            d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.1a2 2 0 0 1-1-1.74v-.51a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"
                        />
                        <circle cx="12" cy="12" r="3" />
                    </svg>
                </a>
            </div>
            <div className="absolute top-2 right-16 z-10">
                <a
                    onClick={lambda -> None { navigate("/");}}
                    className="flex items-center justify-center p-2 rounded-full hover:bg-zinc-800 transition-colors text-zinc-400 hover:text-white bg-zinc-950/50 backdrop-blur-sm cursor-pointer"
                    aria-label="Go Back"
                >
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    >
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="15" y1="9" x2="9" y2="15"></line>
                        <line x1="9" y1="9" x2="15" y2="15"></line>
                    </svg>
                </a>
            </div>
            <aside
                className="w-1/4 border-r border-zinc-900 flex flex-col p-4 space-y-6"
            >
                <div>
                    <h2 className="text-xl font-bold mb-4 text-zinc-100">
                        Operators:
                    </h2>
                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={searchQuery}
                            onChange={lambda e: any  -> None { searchQuery = e.target.value;}}
                            placeholder="Search..."
                            className="flex-1 bg-zinc-900 border border-zinc-800 rounded px-3 py-2 text-sm focus:outline-none focus:border-zinc-700 transition-colors"
                        />
                        <button
                            onClick={lambda -> None { refreshOps();}}
                            className="bg-zinc-800 hover:bg-zinc-700 px-4 py-2 rounded text-sm font-medium transition-colors"
                        >
                            Refresh
                        </button>
                    </div>
                </div>
                <div
                    className="flex-1 overflow-y-auto space-y-2 pr-2 custom-scrollbar"
                >
                    {<p className="text-sm text-zinc-500 px-2">
                        Loading ops...
                    </p>
                    if loading
                    else <p className="text-sm text-zinc-500 px-2">
                        No ops profiled yet. Run the profiling step.
                    </p>
                    if not hasOps
                    else opLinks}
                </div>
                <div className="space-y-4 pt-4 border-t border-zinc-900">
                    <button
                        onClick={lambda -> None { handleProfile();}}
                        className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 rounded shadow-lg shadow-blue-900/20 transition-all"
                    >
                        Profile Model
                    </button>
                    <button
                        onClick={lambda -> None { handleGenerate();}}
                        className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-3 rounded shadow-lg shadow-indigo-900/20 transition-all"
                    >
                        Generate Kernels
                    </button>
                    <button
                        onClick={lambda -> None { handleOptimize();}}
                        className="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-3 rounded shadow-lg shadow-emerald-900/20 transition-all"
                    >
                        Optimize Kernels
                    </button>
                    <button
                        onClick={lambda -> None { handleBenchmark();}}
                        className="w-full bg-amber-500 hover:bg-amber-400 text-white font-bold py-3 rounded shadow-lg shadow-amber-900/20 transition-all"
                    >
                        Benchmark
                    </button>
                    <button
                        onClick={lambda -> None { handleExport();}}
                        className="w-full bg-zinc-800 hover:bg-zinc-700 text-white font-bold py-3 rounded shadow-lg shadow-zinc-900/20 transition-all"
                    >
                        Export .cgins
                    </button>
                    {<button
                        onClick={lambda -> None { handleDownloadExport();}}
                        disabled={downloading}
                        className="w-full bg-zinc-900 hover:bg-zinc-800 text-white font-bold py-3 rounded shadow-lg shadow-zinc-900/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {"Downloading..." if downloading else "Download Export"}
                    </button>
                    if exportPath
                    else <span></span>}
                    {<p className="text-xs text-zinc-500 break-all">
                        {exportPath}
                    </p>
                    if exportPath
                    else <span></span>}
                </div>
            </aside>
            <main className="w-3/4 p-8 overflow-y-auto">
                <h1 className="text-3xl font-bold text-center mb-12 tracking-tight">
                    Project: {name}
                </h1>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
                    <div className="space-y-6">
                        <h3
                            className="text-xl font-semibold border-b border-zinc-800 pb-2"
                        >
                            Model Info:
                        </h3>
                        <div
                            className="h-64 bg-zinc-900/30 rounded-xl border border-zinc-800 p-4 flex flex-col items-center"
                        >
                            <h4 className="text-sm font-medium text-zinc-400 mb-2">
                                PyTorch Operator Usage %
                            </h4>
                            {<p className="text-sm text-zinc-500">
                                No profiling data yet.
                            </p>
                            if not hasPie
                            else <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                    <Pie
                                        data={pieData}
                                        innerRadius={60}
                                        outerRadius={80}
                                        paddingAngle={5}
                                        dataKey="value"
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            "backgroundColor": '#18181b',
                                            "borderColor": '#27272a',
                                            "borderRadius": '8px'
                                        }}
                                        itemStyle={{"color": '#e4e4e7'}}
                                    />
                                    <Legend
                                        layout="vertical"
                                        verticalAlign="middle"
                                        align="right"
                                    />
                                </PieChart>
                            </ResponsiveContainer>}
                        </div>
                        <div
                            className="overflow-hidden rounded-xl border border-zinc-800"
                        >
                            <table className="w-full text-sm text-left">
                                <thead
                                    className="bg-zinc-900 text-zinc-400 font-medium uppercase text-xs"
                                >
                                    <tr>
                                        <th className="px-6 py-3">
                                            Operator
                                        </th>
                                        <th className="px-6 py-3">
                                            Freq
                                        </th>
                                    </tr>
                                </thead>
                                <tbody
                                    className="divide-y divide-zinc-800 bg-zinc-900/50"
                                >
                                    {operatorRows}
                                </tbody>
                            </table>
                        </div>
                        <div
                            className="overflow-hidden rounded-xl border border-zinc-800"
                        >
                            <div
                                className="flex items-center justify-between px-4 py-3 bg-zinc-900 text-zinc-400 text-xs uppercase font-medium"
                            >
                                <span>
                                    Benchmark Results
                                </span>
                                <button
                                    onClick={lambda -> None { refreshBench();}}
                                    className="text-xs text-zinc-300 hover:text-white"
                                >
                                    Refresh
                                </button>
                            </div>
                            {<div
                                className="px-4 py-3 border-b border-zinc-800/60 space-y-1"
                            >
                                <div
                                    className="h-2 w-full rounded bg-zinc-800 overflow-hidden"
                                >
                                    <div
                                        className="h-2 bg-amber-400"
                                        style={{"width": bench_bar_width}}
                                    ></div>
                                </div>
                                <div
                                    className="flex justify-between text-[10px] text-zinc-500"
                                >
                                    <span>
                                        {bench_progress_text}
                                    </span>
                                    <span className="truncate max-w-[140px]">
                                        {bench_message}
                                    </span>
                                </div>
                            </div>
                            if show_bench_progress
                            else <span></span>}
                            {<p className="px-4 py-4 text-sm text-zinc-500">
                                Running benchmark...
                            </p>
                            if benchLoading
                            else <p className="px-4 py-4 text-sm text-zinc-500">
                                No benchmark data yet. Run the Benchmark step.
                            </p>
                            if not hasBench
                            else <table className="w-full text-xs text-left">
                                <thead className="bg-zinc-900/60 text-zinc-500">
                                    <tr>
                                        <th className="px-4 py-2">
                                            Op
                                        </th>
                                        <th className="px-4 py-2">
                                            PyTorch (ms)
                                        </th>
                                        <th className="px-4 py-2">
                                            Kernel (ms)
                                        </th>
                                        <th className="px-4 py-2">
                                            Speedup
                                        </th>
                                        <th className="px-4 py-2">
                                            Source
                                        </th>
                                        <th className="px-4 py-2 text-center">
                                            Export
                                        </th>
                                    </tr>
                                </thead>
                                <tbody
                                    className="divide-y divide-zinc-800 bg-zinc-900/50"
                                >
                                    {benchRows}
                                </tbody>
                            </table>}
                        </div>
                        <p className="text-xs text-zinc-500">
                            Default export includes ops where kernel beats PyTorch.
                        </p>
                        <p className="text-xs text-zinc-500">
                            Kernel errors are shown as "error" (see benchmark log for details).
                        </p>
                    </div>
                    <div className="space-y-6">
                        <h3
                            className="text-xl font-semibold border-b border-zinc-800 pb-2"
                        >
                            Pipeline Status:
                        </h3>
                        <div
                            className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4 space-y-3"
                        >
                            {statusRows}
                            <button
                                onClick={lambda -> None { refreshStatus();}}
                                className="mt-2 w-full bg-zinc-800 hover:bg-zinc-700 text-white font-bold py-2 rounded transition-all"
                            >
                                Refresh Status
                            </button>
                        </div>
                        <div
                            className="h-64 bg-zinc-900/30 rounded-xl border border-zinc-800 p-4 flex flex-col items-center"
                        >
                            <h4 className="text-sm font-medium text-zinc-400 mb-2">
                                Time (ms) vs. Version
                            </h4>
                            {<p className="text-sm text-zinc-500">
                                No optimization data yet.
                            </p>
                            if not hasBar
                            else <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={barData}>
                                    <CartesianGrid
                                        strokeDasharray="3 3"
                                        stroke="#27272a"
                                        vertical={False}
                                    />
                                    <XAxis
                                        dataKey="name"
                                        stroke="#71717a"
                                        tick={{"fill": '#71717a', "fontSize": 12}}
                                        axisLine={False}
                                        tickLine={False}
                                    />
                                    <YAxis
                                        stroke="#71717a"
                                        tick={{"fill": '#71717a', "fontSize": 12}}
                                        axisLine={False}
                                        tickLine={False}
                                    />
                                    <Tooltip
                                        cursor={{"fill": '#27272a'}}
                                        contentStyle={{
                                            "backgroundColor": '#18181b',
                                            "borderColor": '#27272a',
                                            "borderRadius": '8px'
                                        }}
                                        itemStyle={{"color": '#e4e4e7'}}
                                    />
                                    <Bar
                                        dataKey="time"
                                        radius={[4, 4, 0, 0]}
                                        barSize={40}
                                        fill="#3b82f6"
                                    />
                                </BarChart>
                            </ResponsiveContainer>}
                        </div>
                    </div>
                </div>
            </main>
        </div>;
}
