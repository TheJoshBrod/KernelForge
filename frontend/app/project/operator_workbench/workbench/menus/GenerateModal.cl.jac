"""Generate Modal Component"""

import from react { useState, useEffect }
import from "react-router-dom" { useParams }
import from ......components.ui.Icons {
    PlusIcon,
    SettingsIcon,
    CpuIcon,
    BrainIcon,
    ChevronDownIcon
}
import from ......components.ui.SearchableSelect { SearchableSelect }
import from ......components.ui.Modal { Modal }
import from ......context.ConfigContext { useConfig }

def:pub GenerateModal(props: any) -> Any {
    isOpen = props.isOpen;
    onClose = props.onClose;
    onComplete = props.onComplete;
    onStarted = props.onStarted;
    projectName = props.projectName;
    opName = props.opName;
    params = useParams();
    resolvedProjectName = projectName or params.projectName or "";
    routeOpName = params.opName if params and "opName" in params else "";

    config_ctx = useConfig();
    config = config_ctx.config;

    is_working_state = useState(False);
    isWorking = is_working_state[0];
    setIsWorking = is_working_state[1];

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
    selected_gpu_info = config["selected_gpu_info"] if config and "selected_gpu_info" in config else None;
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
                                setLlmProvider(list_models[0]["provider"] if "provider" in list_models[0] else "openai");
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

    def handleModelChange(model_id: str) -> None {
        setLlmModel(model_id);
        for m in models {
            if m["id"] == model_id {
                setLlmProvider(m["provider"] if "provider" in m else "openai");
                break;
            }
        }
    }

    async def handleGenerate -> None {
        if not resolvedProjectName {
            alert("Missing project.");
            return;
        }
        if not selectedOp {
            alert("Select an operator first.");
            return;
        }

        setIsWorking(True);
        try {
            ssh_conns = config["ssh_connections"] if config and "ssh_connections" in config else [];
            active_ssh = config["active_ssh_index"] if config and "active_ssh_index" in config else -1;
            # If the selected GPU is local, do not forward SSH credentials
            # even when active_ssh_index is still set from a previous remote session.
            if selected_gpu_info {
                gpu_conn = selected_gpu_info["connection"] if "connection" in selected_gpu_info else "local";
                if not gpu_conn or gpu_conn == "local" {
                    active_ssh = -1;
                }
            }
            payload = {
                "projectName": resolvedProjectName,
                "ops": [selectedOp],
                "benchmark": True,
                "ssh_connections": ssh_conns,
                "active_ssh_index": active_ssh
            };

            if llmModel {
                payload["llmModel"] = llmModel;
            }
            if llmProvider {
                payload["llmProvider"] = llmProvider;
            }

            resp = await __jacSpawn("StartGenerate", "", payload);
            data = resp.reports[0] if resp and resp.reports else {};
            if data and "success" in data and not data["success"] {
                err = data["error"]
                if "error" in data
                else "Failed to queue generation.";
                alert(err);
            } else {
                onClose();
                if onStarted {
                    onStarted({"_tag": "[GEN]", "_op": selectedOp});
                }
            }
        } except Exception as e {
            console.error("Failed to start generation:", e);
            alert("Failed to start generation.");
        } finally {
            setIsWorking(False);
            if onComplete {
                onComplete();
            }
        }
    }

    if not isOpen {
        return null;
    }

    return (
        <Modal isOpen={isOpen} onClose={onClose} title="Generate New Kernel">
            <div className="p-5 space-y-4">
                <div className="space-y-4">
                    <div className="space-y-1">
                        <label
                            className="text-sm font-medium text-zinc-300 flex items-center gap-2"
                        >
                            <CpuIcon className="w-4 h-4 text-amber-400"/>
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
                                onChange={handleModelChange}
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
                </div>
                <div className="pt-4 flex justify-end gap-3 border-t border-zinc-800">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 text-sm text-zinc-400 hover:text-white transition-colors"
                        disabled={isWorking}
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleGenerate}
                        disabled={isWorking}
                        className={"px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-all shadow-lg hover:shadow-indigo-500/20 flex items-center gap-2 " + (
                            "opacity-50 cursor-not-allowed" if isWorking else ""
                        )}
                    >
                        {"Starting..." if isWorking else "Start Generation"}
                        <PlusIcon
                            className={"w-4 h-4 " + ("opacity-0" if isWorking else "")}
                        />
                    </button>
                </div>
            </div>
        </Modal>
    );
}
