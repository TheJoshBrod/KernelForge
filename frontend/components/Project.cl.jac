"""Project Page Component"""

import from react { useEffect, useState }
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

def:pub Project(
    name: str,
    onNavigate: any,
    getProjectOpsWalker: any,
    getProjectStatusWalker: any,
    startProfileWalker: any,
    startGenerateWalker: any,
    startOptimizeWalker: any,
    startBenchmarkWalker: any,
    exportProjectWalker: any
) -> any {
    COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

    ops_state = useState([]);
    ops = ops_state[0];
    setOps = ops_state[1];

    loading_state = useState(True);
    loading = loading_state[0];
    setLoading = loading_state[1];

    status_state = useState({});
    status = status_state[0];
    setStatus = status_state[1];

    export_state = useState("");
    exportPath = export_state[0];
    setExportPath = export_state[1];

    def format_op(raw: str) -> str {
        prefix = "torch_nn_functional_";
        if raw.startswith(prefix) {
            return "torch.nn.functional." + raw.replace(prefix, "");
        }
        return raw.replace("_", ".");
    }

    async def refreshOps -> None {
        setLoading(True);
        try {
            if getProjectOpsWalker {
                data = await getProjectOpsWalker(name);
                if data and data["ops"] {
                    setOps(data["ops"]);
                } else {
                    setOps([]);
                }
            } else {
                setOps([]);
            }
        } except (error) {
            console.error("Failed to fetch ops:", error);
            setOps([]);
        } finally {
            setLoading(False);
        }
    }

    async def refreshStatus -> None {
        try {
            if getProjectStatusWalker {
                data = await getProjectStatusWalker(name);
                setStatus(data if data else {});
            } else {
                setStatus({});
            }
        } except (error) {
            console.error("Failed to fetch status:", error);
            setStatus({});
        }
    }

    async def refreshAll -> None {
        await refreshOps();
        await refreshStatus();
    }

    useEffect(
        lambda -> None { refreshAll();},
        [name]
    );

    async def handleProfile -> None {
        if startProfileWalker {
            await startProfileWalker(name);
            await refreshStatus();
        }
    }

    async def handleGenerate -> None {
        if startGenerateWalker {
            await startGenerateWalker(name);
            await refreshStatus();
        }
    }

    async def handleOptimize -> None {
        if startOptimizeWalker {
            await startOptimizeWalker(name);
            await refreshStatus();
        }
    }

    async def handleBenchmark -> None {
        if startBenchmarkWalker {
            await startBenchmarkWalker(name);
            await refreshStatus();
        }
    }

    async def handleExport -> None {
        if exportProjectWalker {
            result = await exportProjectWalker(name);
            if result and result["success"] {
                setExportPath(result["path"]);
            }
        }
    }

    operatorFreqData = [];
    for op in ops {
        operatorFreqData.append({
            "operator": format_op(op["name"]),
            "freq": op["count"]
        });
    }

    pieData = [];
    count = 0;
    for op in ops {
        if count >= 4 {
            break;
        }
        pieData.append({
            "name": format_op(op["name"]),
            "value": op["count"],
            "fill": COLORS[0]
        });
        count = count + 1;
    }

    barData = [];

    statusItems = [
        {"key": "profile", "label": "Profile"},
        {"key": "generate", "label": "Generate"},
        {"key": "optimize", "label": "Optimize"},
        {"key": "benchmark", "label": "Benchmark"}
    ];

    statusRows = [];
    for item in statusItems {
        job = status[item["key"]];
        job_status = job["status"] if job and job["status"] else "idle";
        statusRows.append(
            <div
                key={item["key"]}
                className="flex items-center justify-between text-sm text-zinc-300"
            >
                <span>{item["label"]}</span>
                <span className="font-mono text-xs text-zinc-500">
                    {job_status}
                </span>
            </div>
        );
    }

    return
        <div className="flex min-h-screen bg-zinc-950 text-zinc-100 font-sans">
            <div className="absolute top-2 right-6">
                <a
                    onClick={lambda -> None { onNavigate("/settings");}}
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
                    onClick={lambda -> None { onNavigate("/");}}
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
                    if ops.length == 0
                    else ops.map(lambda op: any  -> any { return <a
                        key={op["name"]}
                        onClick={lambda -> None { onNavigate(
                            "/project/" + name + "/" + op["name"]
                        );}}
                        className="block w-full text-left bg-zinc-900/50 hover:bg-zinc-800 border border-zinc-800/50 rounded px-4 py-3 text-sm font-mono text-zinc-300 transition-all cursor-pointer"
                    >
                        {format_op(op["name"])}
                    </a>; })}
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
                            if pieData.length == 0
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
                                    {operatorFreqData.map(lambda row: any  -> any { return <tr
                                        key={row["operator"]}
                                        className="hover:bg-zinc-800/50 transition-colors"
                                    >
                                        <td
                                            className="px-6 py-3 font-mono text-zinc-300"
                                        >
                                            {row["operator"]}
                                        </td>
                                        <td className="px-6 py-3 text-zinc-300">
                                            {row["freq"]}
                                        </td>
                                    </tr>; })}
                                </tbody>
                            </table>
                        </div>
                    </div>
                    <div className="space-y-6">
                        <h3
                            className="text-xl font-semibold border-b border-zinc-800 pb-2"
                        >
                            Pipeline Status:
                        </h3>
                        <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4 space-y-3">
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
                            if barData.length == 0
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
