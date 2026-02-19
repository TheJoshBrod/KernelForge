"""Optimize Modal Component"""

import from react { useState, useEffect }
import from .....components.ui.Icons { ZapIcon, BrainIcon, ActivityIcon, TimerIcon, ChevronDownIcon }
import from .....components.ui.SearchableSelect { SearchableSelect }
import from .....components.ui.Modal { Modal }
import from .....context.ConfigContext { useConfig }

def:pub OptimizeModal(props: any) -> Any {
    isOpen = props.isOpen;
    onClose = props.onClose;
    onComplete = props.onComplete;
    onStarted = props.onStarted;
    projectName = props.projectName;
    opName = props.opName;
    hasTree = props.hasTree if props and "hasTree" in props else True;

    config_ctx = useConfig();
    config = config_ctx.config;

    is_working_state = useState(False);
    isWorking = is_working_state[0];
    setIsWorking = is_working_state[1];

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
    default_provider = provider_options[0]["value"] if provider_options.length > 0 else "openai";

    llm_provider_state = useState(default_provider);
    llmProvider = llm_provider_state[0];
    setLlmProvider = llm_provider_state[1];

    llm_model_state = useState(default_llm);
    llmModel = llm_model_state[0];
    setLlmModel = llm_model_state[1];

    iterations_state = useState(5);
    iterations = iterations_state[0];
    setIterations = iterations_state[1];

    patience_state = useState(2);
    patience = patience_state[0];
    setPatience = patience_state[1];

    # Model List State
    models_state = useState([]);
    models = models_state[0];
    setModels = models_state[1];
    
    loading_models_state = useState(False);
    loadingModels = loading_models_state[0];
    setLoadingModels = loading_models_state[1];

    # Fetch models on open
    useEffect(lambda -> None {
        if isOpen {
            async def fetchModels -> None {
                console.log("Fetching models...");
                setLoadingModels(True);
                try {
                    resp = await __jacSpawn("GetOpenLLMModels", "", {"cfg": config});
                    data = resp.reports[0] if resp and resp.reports else {};
                    if data and "success" in data and data["success"] {
                        if "models" in data {
                            list_models = data["models"];
                            console.log("Fetched models:", list_models);
                            setModels(list_models);
                            # Set default if none selected and list not empty
                            if not llmModel and list_models.length > 0 {
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
        }
    }, [isOpen]);

    provider_option_nodes = [];
    for p in provider_options {
        provider_option_nodes.push(
            <option key={p["value"]} value={p["value"]}>
                {p["label"]}
            </option>
        );
    }

    async def handleOptimize -> None {
        if not hasTree {
            alert("No optimization tree yet for this operator. Generate a new root first.");
            return;
        }
        if not projectName or not opName {
            alert("Missing project or operator.");
            return;
        }

        setIsWorking(True);
        try {
            payload = {
                "projectName": projectName,
                "ops": [opName],
                "iterations": int(iterations),
                "patience": int(patience),
                "minImproveMs": 0.0,
                "minSpeedupPct": 0.0,
                "optimizer_config": {
                    "iterations": int(iterations),
                    "patience": int(patience)
                }
            };

            if llmModel {
                payload["llm_model"] = llmModel;
            }
            if llmProvider {
                payload["llm_provider"] = llmProvider;
            }

            resp = await __jacSpawn(
                "StartOptimizeAdvanced",
                "",
                payload
            );
            data = resp.reports[0] if resp and resp.reports else {};
            if data and "success" in data and not data["success"] {
                err = data["error"]
                if "error" in data
                else "Failed to queue optimization.";
                alert(err);
            } else {
                onClose();
                if onStarted {
                    onStarted(data);
                }
            }
        } except Exception as e {
            console.error("Failed to start optimization:", e);
            alert("Failed to start optimization.");
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
        <Modal isOpen={isOpen} onClose={onClose} title="Optimize Kernel">
            <div className="p-6 space-y-6">
                <div className="space-y-4">
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-zinc-300 flex items-center gap-2">
                            <BrainIcon className="w-4 h-4 text-purple-400" />
                            LLM Model
                        </label>
                        <div className="relative">
                            <SearchableSelect
                                value={llmModel}
                                onChange={setLlmModel}
                                options={models.map(lambda m: dict -> dict {
                                    return {"value": m["id"], "label": m["label"] + " (" + m["provider"] + ")"};
                                })}
                                placeholder="Select a model..."
                                disabled={loadingModels}
                            />
                        </div>
                        {
                            <p className="text-[10px] text-zinc-500">Loading models...</p>
                            if loadingModels 
                            else
                            <p className="text-[10px] text-zinc-500">Select model from {" " + models.length} available</p>
                        }
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                            <label className="text-sm font-medium text-zinc-300 flex items-center gap-2">
                                <ActivityIcon className="w-4 h-4 text-blue-400" />
                                Iterations
                            </label>
                            <input
                                type="number"
                                value={iterations}
                                onChange={lambda e: any -> None { setIterations(e.target.value);}}
                                min="1"
                                max="50"
                                className="w-full bg-zinc-900 border border-zinc-700 rounded-lg p-2.5 text-sm text-zinc-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                            />
                        </div>
                        <div className="space-y-2">
                            <label className="text-sm font-medium text-zinc-300 flex items-center gap-2">
                                <TimerIcon className="w-4 h-4 text-amber-400" />
                                Patience
                            </label>
                            <input
                                type="number"
                                value={patience}
                                onChange={lambda e: any -> None { setPatience(e.target.value);}}
                                min="1"
                                max="10"
                                className="w-full bg-zinc-900 border border-zinc-700 rounded-lg p-2.5 text-sm text-zinc-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                            />
                        </div>
                    </div>
                </div>

                <div className="pt-6 flex justify-end gap-3 border-t border-zinc-800">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 text-sm text-zinc-400 hover:text-white transition-colors"
                        disabled={isWorking}
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleOptimize}
                        disabled={isWorking}
                        className={"px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-all shadow-lg hover:shadow-indigo-500/20 flex items-center gap-2 " + (
                            "opacity-50 cursor-not-allowed" if isWorking else ""
                        )}
                    >
                        {"Starting..." if isWorking else "Start Optimization"}
                        <ZapIcon className={"w-4 h-4 " + ("opacity-0" if isWorking else "")} />
                    </button>
                </div>
            </div>
        </Modal>
    );
}
