"""Optimize Modal Component"""

import from react { useState }
import from .....components.ui.Icons { ZapIcon, BrainIcon, ActivityIcon, TimerIcon }
import from .....components.ui.Modal { Modal }
import from .....context.ConfigContext { useConfig }

def:pub OptimizeModal(props: any) {
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
    default_llm = config["llm_info"]["model"] if config and "llm_info" in config and "model" in config["llm_info"] else "";

    llm_model_state = useState(default_llm);
    llmModel = llm_model_state[0];
    setLlmModel = llm_model_state[1];

    iterations_state = useState(5);
    iterations = iterations_state[0];
    setIterations = iterations_state[1];

    patience_state = useState(2);
    patience = patience_state[0];
    setPatience = patience_state[1];

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
                        <input
                            type="text"
                            value={llmModel}
                            onChange={lambda e: any -> None { setLlmModel(e.target.value);}}
                            placeholder="e.g. claude-3-5-sonnet-20240620"
                            className="w-full bg-zinc-900 border border-zinc-700 rounded-lg p-2.5 text-sm text-zinc-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                        />
                        <p className="text-[10px] text-zinc-500">Override default model from config</p>
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
