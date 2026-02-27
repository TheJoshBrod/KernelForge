"""Cast Export Configuration Modal"""

import from react { useState, useEffect }
import from "react-router-dom" { useParams }
import from ....components.ui.Modal { Modal }
import from ....components.ui.Icons {
    ZapIcon,
    ChevronDownIcon,
    CpuIcon,
    CheckIcon
}
import from ....context.ConfigContext { useConfig }

def:pub CastExportModal(props: any) -> Any {
    isOpen = props.isOpen;
    onClose = props.onClose;
    params = useParams();
    projectName = props.projectName
        or (params.projectName if params and "projectName" in params else "")
        or "";

    config_ctx = useConfig();
    hwInfo = config_ctx.hwInfo if "hwInfo" in config_ctx else {};

    backend_state = useState("CUDA");
    backend = backend_state[0];
    setBackend = backend_state[1];

    ops_state = useState([]);
    ops = ops_state[0];
    setOps = ops_state[1];

    benchmarks_state = useState({});
    benchmarks = benchmarks_state[0];
    setBenchmarks = benchmarks_state[1];

    selected_kernels_state = useState({});
    selectedKernels = selected_kernels_state[0];
    setSelectedKernels = selected_kernels_state[1];

    expanded_op_state = useState("");
    expandedOp = expanded_op_state[0];
    setExpandedOp = expanded_op_state[1];

    op_details_state = useState({});
    opDetails = op_details_state[0];
    setOpDetails = op_details_state[1];

    loading_state = useState(False);
    loading = loading_state[0];
    setLoading = loading_state[1];

    downloading_state = useState(False);
    downloading = downloading_state[0];
    setDownloading = downloading_state[1];

    def to_list(items: Any) -> list {
        if not items {
            return [];
        }
        try {
            if Array.isArray(items) {
                return items;
            }
        } except Exception {
        # no-op
        }
        normalized = [];
        try {
            for item in items {
                normalized.push(item);
            }
            return normalized;
        } except Exception {
        # no-op
        }
        try {
            return Object.values(items);
        } except Exception {
            return [];
        }
    }

    useEffect(
        lambda -> None { if isOpen {
            setOpDetails({});
            setExpandedOp("");
            if projectName {
                async def fetchData -> None {
                    setLoading(True);
                    try {
                        resp_ops = await __jacSpawn(
                            "GetProjectOps", "", {"projectName": projectName}
                        );
                        data_ops = resp_ops.reports[0]
                            if resp_ops and resp_ops.reports
                            else {};
                        op_list = to_list(
                            data_ops["ops"] if data_ops and "ops" in data_ops else []
                        );
                        setOps(op_list);

                        resp_bench = await __jacSpawn(
                            "GetProjectBenchmarks", "", {"projectName": projectName}
                        );
                        data_bench = resp_bench.reports[0]
                            if resp_bench and resp_bench.reports
                            else {};
                        bench_results = to_list(
                            data_bench["results"]
                            if data_bench and "results" in data_bench
                            else []
                        );

                        bench_map = {};
                        init_selection = {};
                        for r in bench_results {
                            if r and "op" in r and r["op"] {
                                bench_map[r["op"]] = r;
                                if "winner" in r and r["winner"] {
                                    w = str(r["winner"]);
                                    w_name = w.split("/").pop();
                                    init_selection[r["op"]] = {
                                        "path": w,
                                        "name": w_name
                                    };
                                }
                            }
                        }
                        setBenchmarks(bench_map);
                        setSelectedKernels(init_selection);
                    } except Exception as e {
                        console.error("Error loading cast config data:", e);
                    } finally {
                        setLoading(False);
                    }
                }
                fetchData();
            }
        }},
        [isOpen, projectName]
    );

    async def handleToggleOp(op_name: str) -> None {
        if expandedOp == op_name {
            setExpandedOp("");
            return;
        }
        setExpandedOp(op_name);
        if op_name in opDetails {
            return;
        }
        try {
            resp = await __jacSpawn(
                "GetOpDetails", "", {"projectName": projectName, "opName": op_name}
            );
            data = resp.reports[0] if resp and resp.reports else {};
            all_kernels = [];
            gen_best = (
                data["generated"]["best"]
                if data
                    and "generated" in data
                    and data["generated"]
                    and "best" in data["generated"]
                    and data["generated"]["best"]
                else {}
            );
            if gen_best and "path" in gen_best and gen_best["path"] {
                all_kernels.push({**gen_best, "label": "Best Generated"});
            }
            opt_kernels = to_list(
                data["optimized"]["kernels"]
                if data
                    and "optimized" in data
                    and data["optimized"]
                    and "kernels" in data["optimized"]
                else []
            );
            for k in opt_kernels {
                all_kernels.push({**k, "label": "Optimized"});
            }
            gen_kernels = to_list(
                data["generated"]["kernels"]
                if data
                    and "generated" in data
                    and data["generated"]
                    and "kernels" in data["generated"]
                else []
            );
            best_path = gen_best["path"] if gen_best and "path" in gen_best else "";
            for k in gen_kernels {
                if not best_path or k["path"] != best_path {
                    all_kernels.push({**k, "label": "Generated"});
                }
            }
            new_details = {**opDetails};
            new_details[op_name] = all_kernels;
            setOpDetails(new_details);
            if not (op_name in selectedKernels) and all_kernels.length > 0 {
                new_sel = {**selectedKernels};
                new_sel[op_name] = {
                    "path": all_kernels[0]["path"],
                    "name": all_kernels[0]["name"]
                };
                setSelectedKernels(new_sel);
            }
        } except Exception as e {
            console.error("Error loading op kernel details:", e);
        }
    }

    def selectKernel(op: str, kp: str, kn: str) -> None {
        new_sel = {**selectedKernels};
        new_sel[op] = {"path": kp, "name": kn};
        setSelectedKernels(new_sel);
    }

    async def handleDownload -> None {
        if not projectName {
            return;
        }
        setDownloading(True);
        try {
            kernel_sel = {};
            for op_name in Object.keys(selectedKernels) {
                entry = selectedKernels[op_name];
                if entry and "path" in entry and entry["path"] {
                    kernel_sel[op_name] = entry["path"];
                }
            }
            resp = await __jacSpawn(
                "DownloadCast",
                "",
                {
                    "projectName": projectName,
                    "backend": backend,
                    "selectedKernels": kernel_sel
                }
            );
            data = resp.reports[0] if resp and resp.reports else {};
            if not data or not ("success" in data) or not data["success"] {
                err = data["error"] if data and "error" in data else "Export failed.";
                alert(err);
                return;
            }
            file_name = data["name"];
            total_size = data["size"] * 1;
            export_path = data["export_path"];
            buffer = Reflect.construct(Uint8Array, [total_size]);
            offset = 0;
            chunk_size = 5242880;
            while offset < total_size {
                chunk_resp = await __jacSpawn(
                    "GetCastChunk",
                    "",
                    {"export_path": export_path, "offset": offset, "chunkSize": chunk_size}
                );
                chunk_data = chunk_resp.reports[0] if chunk_resp and chunk_resp.reports else {};
                if not chunk_data or not ("success" in chunk_data) or not chunk_data["success"] {
                    err = chunk_data["error"] if chunk_data and "error" in chunk_data else "Transfer failed.";
                    alert(err);
                    return;
                }
                bytes_str = atob(chunk_data["data"]);
                chunk_arr = Uint8Array.from(
                    Array.from(bytes_str).map(lambda c: any -> Any { return c.charCodeAt(0); })
                );
                buffer.set(chunk_arr, offset);
                offset = offset + chunk_arr.length;
            }
            blob = Reflect.construct(
                Blob, [[buffer], {"type": "application/octet-stream"}]
            );
            url = URL.createObjectURL(blob);
            a = document.createElement("a");
            a.href = url;
            a.download = file_name;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            onClose();
        } except Exception as e {
            console.error("Download failed:", e);
            alert("Download failed: " + e.toString());
        } finally {
            setDownloading(False);
        }
    }

    gpus = to_list(hwInfo["gpus"] if hwInfo and "gpus" in hwInfo else []);
    gpu_names = [];
    for gpu in gpus {
        if gpu and "name" in gpu and gpu["name"] {
            gpu_names.push(gpu["name"]);
        }
    }
    gpu_label = gpu_names.join(", ") if gpu_names.length > 0 else "Unknown GPU";

    op_rows = [];
    for op in to_list(ops) {
        op_name = op["name"];
        bench = benchmarks[op_name] if op_name in benchmarks else {};

        pytorch_ms = None;
        forged_ms = None;
        try {
            if bench and "pytorch_ms" in bench and bench["pytorch_ms"] {
                pytorch_ms = bench["pytorch_ms"] * 1;
            }
        } except Exception {
        # no-op
        }
        try {
            if bench and "forged_ms" in bench and bench["forged_ms"] {
                forged_ms = bench["forged_ms"] * 1;
            }
        } except Exception {
        # no-op
        }
        speedup = None;
        if pytorch_ms and forged_ms and forged_ms > 0 {
            speedup = (pytorch_ms / forged_ms).toFixed(2);
        }
        has_kernel = bench and "winner" in bench and bench["winner"];
        sel = selectedKernels[op_name] if op_name in selectedKernels else {};
        sel_name = (
            sel["name"]
            if sel and "name" in sel and sel["name"]
            else "default best"
        );
        is_expanded = expandedOp == op_name;
        kernels_for_op_list = to_list(
            opDetails[op_name] if op_name in opDetails else []
        );

        speedup_badge_nodes = [];
        if speedup {
            speedup_badge_nodes.push(
                <span
                    className="shrink-0 text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-900/40 text-emerald-400 border border-emerald-800/50"
                >
                    {speedup}x
                </span>
            );
        } else {
            speedup_badge_nodes.push(
                <span
                    className={"shrink-0 text-xs px-2 py-0.5 rounded-full border " + (
                        "bg-indigo-900/30 text-indigo-400 border-indigo-800/50"
                        if has_kernel
                        else "bg-zinc-800/50 text-zinc-500 border-zinc-700"
                    )}
                >
                    {"has kernel" if has_kernel else "no kernel"}
                </span>
            );
        }

        kernel_items = [];
        for k in kernels_for_op_list {
            k_path = k["path"] if "path" in k else "";
            k_name_k = k["name"] if "name" in k else k_path;
            k_label = k["label"] if "label" in k else "";
            k_size = k["size"] if "size" in k else 0;
            is_sel = (
                op_name in selectedKernels
                and selectedKernels[op_name]
                and "path" in selectedKernels[op_name]
                and selectedKernels[op_name]["path"] == k_path
            );
            size_str = " · " + str(k_size) + " B" if k_size else "";

            radio_fill_nodes = [];
            if is_sel {
                radio_fill_nodes.push(
                    <div className="w-1.5 h-1.5 rounded-full bg-indigo-400"/>
                );
            }

            check_icon_nodes = [];
            if is_sel {
                check_icon_nodes.push(
                    <CheckIcon className="w-3.5 h-3.5 text-indigo-400 shrink-0"/>
                );
            }

            kernel_items.push(
                <button
                    key={k_path}
                    onClick={lambda -> None { selectKernel(op_name, k_path, k_name_k);}}
                    className={"w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-colors " + (
                        "bg-indigo-900/30 border border-indigo-700/40"
                        if is_sel
                        else "hover:bg-zinc-800"
                    )}
                >
                    <div
                        className={"w-3.5 h-3.5 rounded-full border-2 shrink-0 flex items-center justify-center " + (
                            "border-indigo-500" if is_sel else "border-zinc-600"
                        )}
                    >
                        {radio_fill_nodes}
                    </div>
                    <div className="flex-1 min-w-0">
                        <p
                            className={"text-xs font-mono truncate " + (
                                "text-indigo-200" if is_sel else "text-zinc-400"
                            )}
                        >
                            {k_name_k}
                        </p>
                        <p className="text-[10px] text-zinc-600 mt-0.5">
                            {k_label + size_str}
                        </p>
                    </div>
                    {check_icon_nodes}
                </button>
            );
        }

        expanded_panel_nodes = [];
        if is_expanded {
            empty_msg = (
                "Loading kernels..."
                if not (op_name in opDetails)
                else "No kernel files found."
            );
            expanded_panel_nodes.push(
                <div className="border-t border-zinc-800 bg-zinc-950 px-4 py-3 space-y-1">
                    {<div className="text-xs text-zinc-500 py-2 text-center">
                        {empty_msg}
                    </div>
                    if kernel_items.length == 0
                    else kernel_items}
                </div>
            );
        }

        op_rows.push(
            <div
                key={op_name}
                className="rounded-lg border border-zinc-800 overflow-hidden"
            >
                <button
                    onClick={lambda -> None { handleToggleOp(op_name);}}
                    className="w-full flex items-center gap-3 px-4 py-3 bg-zinc-900 hover:bg-zinc-800 transition-colors text-left"
                >
                    <span className="flex-1 font-mono text-sm text-zinc-200 truncate">
                        {op_name}
                    </span>
                    {speedup_badge_nodes}
                    <span
                        className="shrink-0 text-[11px] text-zinc-500 max-w-[110px] truncate hidden sm:block"
                    >
                        {sel_name}
                    </span>
                    <ChevronDownIcon
                        className={"w-4 h-4 text-zinc-500 transition-transform duration-200 shrink-0 " + (
                            "rotate-180" if is_expanded else ""
                        )}
                    />
                </button>
                {expanded_panel_nodes}
            </div>
        );
    }

    btn_icon_nodes = [];
    if downloading {
        btn_icon_nodes.push(
            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                />
                <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
            </svg>
        );
    } else {
        btn_icon_nodes.push(<ZapIcon className="w-4 h-4"/>);
    }

    if not isOpen {
        return null;
    }

    return (
        <Modal isOpen={isOpen} onClose={onClose} title="Configure Cast Export">
            <div className="p-6 space-y-6">
                <div
                    className="flex items-center gap-3 bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-3"
                >
                    <CpuIcon className="w-4 h-4 text-indigo-400 shrink-0"/>
                    <div>
                        <p
                            className="text-[10px] text-zinc-500 leading-none mb-0.5 uppercase tracking-wide"
                        >
                            Optimized for
                        </p>
                        <p className="text-sm font-medium text-zinc-200">
                            {gpu_label}
                        </p>
                    </div>
                </div>
                <div className="space-y-2">
                    <label
                        className="text-sm font-medium text-zinc-300 flex items-center gap-2"
                    >
                        <ZapIcon className="w-4 h-4 text-[var(--accent)]"/>
                        Kernel Backend
                    </label>
                    <div className="relative">
                        <select
                            value={backend}
                            onChange={lambda e: any -> None { setBackend(e.target.value);}}
                            className="w-full bg-zinc-900 border border-zinc-700 rounded-lg p-2.5 text-sm text-zinc-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 appearance-none"
                        >
                            <option value="CUDA">
                                CUDA
                            </option>
                        </select>
                        <ChevronDownIcon
                            className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 w-4 h-4 pointer-events-none"
                        />
                    </div>
                    <p className="text-[10px] text-zinc-500 px-1">
                        Additional backends coming soon.
                    </p>
                </div>
                <div className="space-y-2">
                    <label className="text-sm font-medium text-zinc-300">
                        Operator Kernels
                    </label>
                    {<div
                        className="flex items-center justify-center py-8 text-zinc-500 text-sm"
                    >
                        Loading operators...
                    </div>
                    if loading
                    else <div className="space-y-1 max-h-72 overflow-y-auto pr-1">
                        {<p className="text-sm text-zinc-500 py-4 text-center">
                            No profiled operators found.
                        </p>
                        if op_rows.length == 0
                        else op_rows}
                    </div>}
                </div>
                <div className="pt-6 flex justify-end gap-3 border-t border-zinc-800">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 text-sm text-zinc-400 hover:text-white transition-colors"
                        disabled={downloading}
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleDownload}
                        disabled={downloading}
                        className={"px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-all shadow-lg hover:shadow-indigo-500/20 flex items-center gap-2 " + (
                            "opacity-50 cursor-not-allowed" if downloading else ""
                        )}
                    >
                        {btn_icon_nodes}
                        {"Downloading..." if downloading else "Download Cast"}
                    </button>
                </div>
            </div>
        </Modal>
    );
}
