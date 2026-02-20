"""Upload Menu Component"""

import from react { useState, useRef, useEffect }
import from "react-router-dom" { useParams }
import from .....components.ui.Icons { UploadIcon, CodeIcon, SettingsIcon }
import from .....components.ui.Modal { Modal }

import from .....context.ConfigContext { useConfig }

def:pub UploadModal(props: any) -> Any {
    isOpen = props.isOpen;
    onClose = props.onClose;
    onSuccess = props.onSuccess;
    projectName = props.projectName;
    opName = props.opName;
    params = useParams();
    resolvedProjectName = projectName or params.projectName or "";
    routeOpName = params.opName if params and "opName" in params else "";

    # Context
    config_ctx = useConfig();
    hwInfo = config_ctx.hwInfo if "hwInfo" in config_ctx else {};

    # State
    code_state = useState("");
    code = code_state[0];
    setCode = code_state[1];

    backend_state = useState("CUDA");
    backend = backend_state[0];
    setBackend = backend_state[1];

    is_loading_state = useState(False);
    isLoading = is_loading_state[0];
    setIsLoading = is_loading_state[1];

    selected_op_state = useState(opName or routeOpName or "");
    selectedOp = selected_op_state[0];
    setSelectedOp = selected_op_state[1];

    ops_state = useState([]);
    ops = ops_state[0];
    setOps = ops_state[1];

    loading_ops_state = useState(False);
    loadingOps = loading_ops_state[0];
    setLoadingOps = loading_ops_state[1];

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

    def list_len(items: Any) -> int {
        return to_list(items).length;
    }

    opLocked = False;
    if opName or routeOpName {
        opLocked = True;
    }

    # Determine available backends
    available_devices = [{"label": "Triton (Python)", "value": "Triton"}];
    if hwInfo and "cuda_available" in hwInfo and hwInfo["cuda_available"] {
        available_devices.splice(0, 0, {"label": "CUDA", "value": "CUDA"});
    } elif hwInfo and "mps_available" in hwInfo and hwInfo["mps_available"] {
        available_devices.splice(0, 0, {"label": "MPS", "value": "MPS"});
    }

    # Update default backend
    useEffect(
        lambda -> None { current_valid = False;for dev in available_devices {
            if dev["value"] == backend {
                current_valid = True;
                break;
            }
        }if not current_valid and available_devices.length > 0 {
            setBackend(available_devices[0]["value"]);
        }},
        [hwInfo, backend]
    );

    useEffect(
        lambda -> None { if isOpen {
            if opName or routeOpName {
                setSelectedOp(opName or routeOpName or "");
            }
        }},
        [isOpen, opName, routeOpName]
    );

    useEffect(
        lambda -> None { if isOpen and resolvedProjectName {
            async def fetchOps -> None {
                setLoadingOps(True);
                try {
                    resp = await __jacSpawn(
                        "GetProjectOps", "", {"projectName": resolvedProjectName}
                    );
                    data = resp.reports[0] if resp and resp.reports else {};
                    op_list = to_list(data["ops"] if data and "ops" in data else []);
                    setOps(op_list);
                    if (not selectedOp) and list_len(op_list) > 0 {
                        setSelectedOp(op_list[0]["name"]);
                    }
                } except Exception as e {
                    console.error("Error fetching operators:", e);
                    setOps([]);
                } finally {
                    setLoadingOps(False);
                }
            }
            fetchOps();
        }},
        [isOpen, resolvedProjectName]
    );

    file_input_ref = useRef(None);

    # Handlers
    def handleFileSelect(e: any) {
        file = e.target.files[0];
        if file {
            reader = Reflect.construct(FileReader, []);
            def on_load(evt: any) {
                setCode(evt.target.result);
            }
            reader.onload = on_load;
            reader.readAsText(file);
        }
    }

    async def handleCreate -> None {
        if not code {
            alert("Please enter some code or upload a file.");
            return;
        }
        if not resolvedProjectName {
            alert("Missing project.");
            return;
        }
        if not selectedOp {
            alert("Select an operator first.");
            return;
        }

        setIsLoading(True);
        try {
            # Call the ImportKernel walker with the code and backend
            resp = await __jacSpawn(
                "ImportKernel",
                "",
                {
                    "projectName": resolvedProjectName,
                    "opName": selectedOp,
                    "source": code,
                    "backend": backend
                }
            );
            data = resp.reports[0] if resp and resp.reports else {};
            if data and "success" in data and not data["success"] {
                err = data["error"] if "error" in data else "Failed to create kernel.";
                alert(err);
                return;
            }

            alert("Kernel created successfully!");

            onClose();
            if onSuccess {
                onSuccess();
            }

            # Reset state
            setCode("");
            setBackend("CUDA");
        } except Exception as e {
            console.error("Upload failed", e);
            alert("Failed to create kernel: " + e.toString());
        } finally {
            setIsLoading(False);
        }
    }

    if not isOpen {
        return null;
    }

    return (
        <Modal isOpen={isOpen} onClose={onClose} title="Upload Custom Kernel">
            <div className="p-6 space-y-6">
                <div className="space-y-4">
                    <div className="space-y-2">
                        <label
                            className="text-sm font-medium text-zinc-300 flex items-center gap-2"
                        >
                            <SettingsIcon className="w-4 h-4 text-amber-400"/>
                            Operator
                        </label>
                        {<div
                            className="w-full bg-zinc-900 border border-zinc-700 rounded-lg p-2.5 text-sm text-zinc-200 font-mono"
                        >
                            {selectedOp or "(missing operator context)"}
                        </div>
                        if opLocked
                        else <select
                            value={selectedOp}
                            onChange={lambda e: any  -> None { setSelectedOp(
                                e.target.value
                            );}}
                            className="w-full bg-zinc-900 border border-zinc-700 rounded-lg p-2.5 text-sm text-zinc-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                            disabled={loadingOps}
                        >
                            {<option value="">
                                {"Loading operators..."
                                if loadingOps
                                else "Select operator"}
                            </option>
                            if not selectedOp
                            else <span/>}{ops.map(
                                lambda item: dict  -> Any { return
                                    <option key={item["name"]} value={item["name"]}>
                                        {item["name"]}
                                    </option>; }
                            )}
                        </select>}
                        <p className="text-[10px] text-zinc-500">
                            {resolvedProjectName
                            if resolvedProjectName
                            else "Project missing from route context."}
                        </p>
                    </div>
                    <div className="flex items-center justify-between">
                        <label
                            className="text-sm font-medium text-zinc-300 flex items-center gap-2"
                        >
                            <CodeIcon className="w-4 h-4 text-indigo-400"/>
                            Kernel Source Code
                        </label>
                        <div>
                            <input
                                type="file"
                                ref={file_input_ref}
                                className="hidden"
                                onChange={lambda e: any  -> None { handleFileSelect(e);}}
                                accept=".cu,.cpp,.c,.py,.jac"
                            />
                            <button
                                onClick={lambda -> None { file_input_ref.current.click();}}
                                className="text-xs flex items-center gap-2 px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 hover:text-white rounded-md text-zinc-400 transition-colors border border-zinc-700"
                            >
                                <UploadIcon className="w-3 h-3"/>
                                Load from File
                            </button>
                        </div>
                    </div>
                    <div className="relative group">
                        <textarea
                            value={code}
                            onChange={lambda e: any  -> None { setCode(e.target.value);}}
                            className="w-full h-80 bg-zinc-950 border border-zinc-800 rounded-lg p-4 font-mono text-xs text-zinc-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 resize-none leading-relaxed"
                            placeholder="// Enter your kernel code here (e.g. CUDA C++)..."
                            spellCheck={False}
                        />
                        <div
                            className="absolute bottom-4 right-4 text-[10px] text-zinc-600 pointer-events-none"
                        >
                            {code.length} chars
                        </div>
                    </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                        <label
                            className="text-sm font-medium text-zinc-300 flex items-center gap-2"
                        >
                            <SettingsIcon className="w-4 h-4 text-amber-400"/>
                            Target Backend
                        </label>
                        <div className="relative">
                            <select
                                value={backend}
                                onChange={lambda e: any  -> None { setBackend(
                                    e.target.value
                                );}}
                                className="w-full bg-zinc-900 border border-zinc-700 rounded-lg p-2.5 text-sm text-zinc-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 appearance-none"
                            >
                                {available_devices.map(
                                    lambda d: dict  -> Any { return
                                        <option key={d["value"]} value={d["value"]}>
                                            {d["label"]}
                                        </option>; }
                                )}
                            </select>
                            <div
                                className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none"
                            >
                                <svg
                                    className="w-4 h-4 text-zinc-500"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                >
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth="2"
                                        d="M19 9l-7 7-7-7"
                                    />
                                </svg>
                            </div>
                        </div>
                        <p className="text-[10px] text-zinc-500 px-1">
                            Select the backend compatible with your kernel code.
                        </p>
                    </div>
                </div>
                <div className="pt-6 flex justify-end gap-3 border-t border-zinc-800">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 text-sm text-zinc-400 hover:text-white transition-colors"
                        disabled={isLoading}
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleCreate}
                        disabled={isLoading}
                        className={"px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-all shadow-lg hover:shadow-indigo-500/20 flex items-center gap-2 " + (
                            "opacity-50 cursor-not-allowed" if isLoading else ""
                        )}
                    >
                        {"Creating..." if isLoading else "Create Kernel"}
                        <UploadIcon
                            className={"w-4 h-4 " + ("opacity-0" if isLoading else "")}
                        />
                    </button>
                </div>
            </div>
        </Modal>
    );
}
