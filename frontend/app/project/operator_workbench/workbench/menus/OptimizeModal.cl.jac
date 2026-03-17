"""Optimize Modal Component"""

import from react { useState, useEffect }
import from "react-router-dom" { useParams }
import from ......components.ui.Icons {
    ZapIcon,
    BrainIcon,
    ActivityIcon,
    ChevronDownIcon,
    SettingsIcon
}
import from ......components.ui.SearchableSelect { SearchableSelect }
import from ......components.ui.Modal { Modal }
import from ......context.ConfigContext { useConfig }
import from sonner { toast }

def:pub OptimizeModal(props: any) -> Any {
    isOpen = props.isOpen;
    onClose = props.onClose;
    onComplete = props.onComplete;
    onStarted = props.onStarted;
    projectName = props.projectName;
    opName = props.opName;
    modalSession = props.modalSession if props and "modalSession" in props else None;
    hasTree = props.hasTree if props and "hasTree" in props else True;
    hasProjectWork = bool(props.hasProjectWork) if props and "hasProjectWork" in props and props.hasProjectWork else False;
    params = useParams();
    resolvedProjectName = projectName or params.projectName or "";
    routeOpName = params.opName if params and "opName" in params else "";

    config_ctx = useConfig();
    config = config_ctx.config;

    ssh_connections = config["ssh_connections"] if config and "ssh_connections" in config else [];
    active_ssh_index = config["active_ssh_index"] if config and "active_ssh_index" in config else -1;
    
    selected_gpu_info = config["selected_gpu_info"] if config and "selected_gpu_info" in config else None;
    is_remote_gpu = selected_gpu_info and "connection" in selected_gpu_info and selected_gpu_info["connection"] != "local";
    
    active_ssh = None;
    if is_remote_gpu and active_ssh_index >= 0 and ssh_connections.length > active_ssh_index {
        active_ssh = ssh_connections[active_ssh_index];
    }

    is_working_state = useState(False);
    isWorking = is_working_state[0];
    setIsWorking = is_working_state[1];
    submit_armed_state = useState(False);
    submitArmed = submit_armed_state[0];
    setSubmitArmed = submit_armed_state[1];

    selected_op_state = useState(opName or routeOpName or "");
    selectedOp = selected_op_state[0];
    setSelectedOp = selected_op_state[1];

    ops_state = useState([]);
    ops = ops_state[0];
    setOps = ops_state[1];

    loading_ops_state = useState(False);
    loadingOps = loading_ops_state[0];
    setLoadingOps = loading_ops_state[1];

    opLocked = False;
    if opName or routeOpName {
        opLocked = True;
    }

    # Config State
    default_llm = "";
    api_keys = config["api_keys"] if config and "api_keys" in config else {};
    provider_options = [];
    if api_keys and "openai" in api_keys and api_keys["openai"] {
        provider_options.push({"label": "OpenAI", "value": "openai"});
    }
    if api_keys and "anthropic" in api_keys and api_keys["anthropic"] {
        provider_options.push({"label": "Anthropic", "value": "anthropic"});
    }
    if api_keys and "google" in api_keys and api_keys["google"] {
        provider_options.push({"label": "Google", "value": "google"});
    }
    if provider_options.length == 0 {
        provider_options = [
            {"label": "OpenAI", "value": "openai"},
            {"label": "Anthropic", "value": "anthropic"},
            {"label": "Google", "value": "google"}
        ];
    }
    default_provider = provider_options[0]["value"]
    if provider_options.length > 0
    else "openai";

    llm_provider_state = useState(default_provider);
    llmProvider = llm_provider_state[0];
    setLlmProvider = llm_provider_state[1];

    llm_model_state = useState(default_llm);
    llmModel = llm_model_state[0];
    setLlmModel = llm_model_state[1];

    iterations_state = useState(5);
    iterations = iterations_state[0];
    setIterations = iterations_state[1];

    workers_state = useState(4);
    workers = workers_state[0];
    setWorkers = workers_state[1];

    # Model List State
    models_state = useState([]);
    models = models_state[0];
    setModels = models_state[1];

    loading_models_state = useState(False);
    loadingModels = loading_models_state[0];
    setLoadingModels = loading_models_state[1];

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

    # Fetch models on open
    useEffect(
        lambda -> None { if isOpen {
            async def fetchModels -> None {
                console.log("Fetching models...");
                setLoadingModels(True);
                try {
                    resp = await __jacSpawn("GetOpenLLMModels", "", {"cfg": config});
                    data = resp.reports[0] if resp and resp.reports else {};
                    if data and "success" in data and data["success"] {
                        if "models" in data {
                            list_models = to_list(data["models"]);
                            console.log("Fetched models:", list_models);
                            setModels(list_models);
                            # Set default if none selected and list not empty
                            if not llmModel and list_len(list_models) > 0 {
                                setLlmModel(list_models[0]["id"]);
                            }
                        }
                    }
                } except Exception as e {
                    console.error("Error fetching models:", e);
                } finally {
                    setLoadingModels(False);
                }
            }
            fetchModels();
        }},
        [isOpen]
    );

    useEffect(
        lambda -> Any { arm_timer = None;if isOpen {
            setIsWorking(False);
            setSubmitArmed(False);
            if opName or routeOpName {
                setSelectedOp(opName or routeOpName or "");
            }
            arm_timer = setTimeout(lambda -> None { setSubmitArmed(True);}, 200);
        } else {
            setSubmitArmed(False);
        }return lambda -> None { if arm_timer { clearTimeout(arm_timer); }};},
        [isOpen, opName, routeOpName]
    );

    def close_modal -> None {
        if onClose {
            onClose(modalSession);
        }
    }

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
                    if (not opName) and (not routeOpName) and list_len(op_list) > 0 {
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

    provider_option_nodes = [];
    for p in provider_options {
        provider_option_nodes.push(
            <option key={p["value"]} value={p["value"]}>
                {p["label"]}
            </option>
        );
    }

    async def handleOptimize -> None {
        if not submitArmed {
            return;
        }
        if not hasTree and selectedOp == (opName or routeOpName or "") {
            alert(
                "No optimization tree yet for this operator. Generate a new root first."
            );
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

        setIsWorking(True);
        submitId = "";
        try {
            submitId = "opt-" + str(Date.now()) + "-" + selectedOp;
            optimisticQueued = bool(hasProjectWork);
            if onStarted {
                onStarted({
                    "_tag": "[OPT]",
                    "_op": selectedOp,
                    "_workers": int(workers),
                    "_queued": optimisticQueued,
                    "_optimistic": True,
                    "_submit_id": submitId
                });
            }
            payload = {
                "projectName": resolvedProjectName,
                "ops": [selectedOp],
                "iterations": int(iterations),
                "patience": 2,
                "minImproveMs": 0.0,
                "minSpeedupPct": 0.0,
                "workers": int(workers),
                "optimizer_config": {
                    "iterations": int(iterations),
                    "patience": 2
                }
            };

            if llmModel {
                payload["llm_model"] = llmModel;
            }
            if llmProvider {
                payload["llm_provider"] = llmProvider;
            }
            if active_ssh {
                payload["ssh_connections"] = ssh_connections;
                payload["active_ssh_index"] = active_ssh_index;
            }

            resp = await __jacSpawn("StartOptimize", "", payload);
            data = resp.reports[0] if resp and resp.reports else {};
            if data and "success" in data and not data["success"] {
                if onStarted and submitId {
                    onStarted({"_rollback_submit_id": submitId});
                }
                err = data["error"]
                if "error" in data
                else "Failed to queue optimization.";
                alert(err);
            } elif data and "queued" in data and data["queued"] {
                if data and "gpu_warning" in data and data["gpu_warning"] {
                    toast.warning(data["gpu_warning"]);
                }
                if onStarted {
                    onStarted({
                        "_tag": "[OPT]",
                        "_op": selectedOp,
                        "_workers": int(workers),
                        "_queued": True,
                        "_submit_id": submitId,
                        "_message": (
                            data["message"]
                            if "message" in data and data["message"]
                            else "Optimization queued."
                        )
                    });
                }
                close_modal();
            } elif data and "skipped" in data and data["skipped"] {
                if onStarted and submitId {
                    onStarted({"_rollback_submit_id": submitId});
                }
                msg = data["message"]
                if "message" in data
                else "Optimization already in progress.";
                alert(msg);
            } else {
                if onStarted {
                    onStarted({"_tag": "[OPT]", "_op": selectedOp, "_workers": int(workers), "_submit_id": submitId});
                }
                close_modal();
            }
        } except Exception as e {
            if onStarted and submitId {
                onStarted({"_rollback_submit_id": submitId});
            }
            console.error("Failed to start optimization:", e);
            alert("Failed to start optimization.");
        } finally {
            setIsWorking(False);
        }
    }

    if not isOpen {
        return null;
    }

    return (
        <Modal isOpen={isOpen} onClose={close_modal} title="Optimize Kernel">
            <div className="p-6 space-y-6">
                <div className="space-y-4">
                    <div className="space-y-1">
                        <label
                            className="text-sm font-medium text-zinc-300 flex items-center gap-2"
                        >
                            <ZapIcon className="w-4 h-4 text-amber-400"/>
                            Operator
                        </label>
                        {<div
                            className="w-full bg-zinc-900 border border-zinc-700 rounded-lg p-2.5 text-sm text-zinc-200 font-mono"
                        >
                            {selectedOp or "(missing operator context)"}
                        </div>
                        if opLocked
                        else <div className="relative">
                            <select
                                value={selectedOp}
                                onChange={lambda e: any  -> None { setSelectedOp(
                                    e.target.value
                                );}}
                                className="w-full bg-zinc-900 border border-zinc-700 rounded-lg p-2.5 text-sm text-zinc-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 appearance-none"
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
                                        <option
                                            key={item["name"]}
                                            value={item["name"]}
                                        >
                                            {item["name"]}
                                        </option>; }
                                )}
                            </select>
                            <ChevronDownIcon
                                className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 w-4 h-4 pointer-events-none"
                            />
                        </div>}
                        <p className="text-[10px] text-zinc-500">
                            {resolvedProjectName
                            if resolvedProjectName
                            else "Project missing from route context."}
                        </p>
                    </div>
                    <div className="space-y-1">
                        <label
                            className="text-sm font-medium text-zinc-300 flex items-center gap-2"
                        >
                            <BrainIcon className="w-4 h-4 text-purple-400"/>
                            LLM Model
                        </label>
                        <div className="relative">
                            <SearchableSelect
                                value={llmModel}
                                onChange={setLlmModel}
                                options={models.map(
                                    lambda m: dict  -> dict { return {
                                        "value": m["id"],
                                        "label": m["label"] + " (" + m["provider"] + ")"
                                    }; }
                                )}
                                placeholder="Select a model..."
                                disabled={loadingModels}
                            />
                        </div>
                        {<p className="text-[10px] text-zinc-500">
                            Loading models...
                        </p>
                        if loadingModels
                        else <p className="text-[10px] text-zinc-500">
                            Select model from{" " + str(list_len(models))} available
                        </p>}
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                            <label
                                className="text-sm font-medium text-zinc-300 flex items-center gap-2"
                            >
                                <ActivityIcon className="w-4 h-4 text-blue-400"/>
                                Iterations
                            </label>
                            <input
                                type="number"
                                value={iterations}
                                onChange={lambda e: any  -> None { setIterations(
                                    e.target.value
                                );}}
                                min="1"
                                max="50"
                                className="w-full bg-zinc-900 border border-zinc-700 rounded-lg p-2.5 text-sm text-zinc-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                            />
                        </div>
                        <div className="space-y-2">
                            <label
                                className="text-sm font-medium text-zinc-300 flex items-center gap-2"
                            >
                                <SettingsIcon className="w-4 h-4 text-purple-400"/>
                                Parallel Workers
                            </label>
                            <input
                                type="number"
                                value={workers}
                                onChange={lambda e: any -> None { setWorkers(e.target.value); }}
                                min="1"
                                max="32"
                                className="w-full bg-zinc-900 border border-zinc-700 rounded-lg p-2.5 text-sm text-zinc-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                            />
                        </div>
                    </div>
                </div>
                {<div className="flex items-center gap-2 px-3 py-2 bg-emerald-950/40 border border-emerald-800/40 rounded-lg text-xs text-emerald-400">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0"/>
                    {"Remote: " + str(active_ssh["user"]) + "@" + str(active_ssh["host"]) if active_ssh and "user" in active_ssh and "host" in active_ssh else "Remote GPU"}
                </div>
                if active_ssh
                else <span/>}
                <div className="pt-4 flex justify-end gap-3 border-t border-zinc-800">
                    <button
                        onClick={close_modal}
                        className="px-4 py-2 text-sm text-zinc-400 hover:text-white transition-colors"
                        disabled={isWorking}
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleOptimize}
                        disabled={isWorking or not submitArmed}
                        className={"px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-all shadow-lg hover:shadow-indigo-500/20 flex items-center gap-2 " + (
                            "opacity-50 cursor-not-allowed"
                            if isWorking or not submitArmed
                            else ""
                        )}
                    >
                        {
                            "Starting..."
                            if isWorking
                            else ("Preparing..." if not submitArmed else "Start Optimization")
                        }
                        <ZapIcon
                            className={"w-4 h-4 " + ("opacity-0" if isWorking or not submitArmed else "")}
                        />
                    </button>
                </div>
            </div>
        </Modal>
    );
}
