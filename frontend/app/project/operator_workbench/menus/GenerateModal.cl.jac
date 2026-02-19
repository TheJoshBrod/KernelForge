"""Generate Modal Component"""

import from react { useState, useEffect }
import from .....components.ui.Icons { PlusIcon, SettingsIcon, CpuIcon, BrainIcon, ActivityIcon }
import from .....components.ui.Modal { Modal }
import from .....context.ConfigContext { useConfig }

def:pub GenerateModal(props: any) {
    isOpen = props.isOpen;
    onClose = props.onClose;
    onComplete = props.onComplete;
    onStarted = props.onStarted;
    projectName = props.projectName;
    opName = props.opName;

    config_ctx = useConfig();
    config = config_ctx.config;

    is_working_state = useState(False);
    isWorking = is_working_state[0];
    setIsWorking = is_working_state[1];

    # Config State
    # Config State
    default_llm = "";
    
    llm_model_state = useState(default_llm);
    llmModel = llm_model_state[0];
    setLlmModel = llm_model_state[1];

    max_attempts_state = useState(8);
    maxAttempts = max_attempts_state[0];
    setMaxAttempts = max_attempts_state[1];

    target_device_state = useState("cuda");
    targetDevice = target_device_state[0];
    setTargetDevice = target_device_state[1];

    use_template_state = useState(True);
    useTemplate = use_template_state[0];
    setUseTemplate = use_template_state[1];

    # Determine available devices
    hwInfo = config_ctx.hwInfo if "hwInfo" in config_ctx else {};
    available_devices = [{"label": "Triton", "value": "triton"}];
    
    if hwInfo and "cuda_available" in hwInfo and hwInfo["cuda_available"] {
        available_devices.splice(0, 0, {"label": "CUDA", "value": "cuda"});
    } elif hwInfo and "mps_available" in hwInfo and hwInfo["mps_available"] {
        available_devices.splice(0, 0, {"label": "MPS", "value": "mps"});
    }
    
    # Update default target device if current selection is invalid
    useEffect(lambda -> None {
        current_valid = False;
        for dev in available_devices {
            if dev["value"] == targetDevice {
                current_valid = True;
                break;
            }
        }
        if not current_valid and available_devices.length > 0 {
            setTargetDevice(available_devices[0]["value"]);
        }
    }, [hwInfo, targetDevice]);

    async def handleGenerate -> None {
        if not projectName or not opName {
            alert("Missing project or operator.");
            return;
        }

        setIsWorking(True);
        try {
            payload = {
                "projectName": projectName,
                "ops": [opName],
                "optimize": False,
                "benchmark": False,
                "generator_config": {
                    "max_attempts": int(maxAttempts),
                    "target_device": targetDevice,
                    "use_baseline_kernels": True,
                    "use_baseline_as_template": useTemplate,
                    "extra_validation_cases": 0
                }
            };
            
            if llmModel {
                payload["llm_model"] = llmModel;
            }

            resp = await __jacSpawn(
                "StartGenerateAdvanced",
                "",
                payload
            );
            data = resp.reports[0] if resp and resp.reports else {};
            if data and "success" in data and not data["success"] {
                err = data["error"]
                if "error" in data
                else "Failed to queue generation.";
                alert(err);
            } else {
                onClose();
                if onStarted {
                    onStarted(data);
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
                                Max Attempts
                            </label>
                            <input
                                type="number"
                                value={maxAttempts}
                                onChange={lambda e: any -> None { setMaxAttempts(e.target.value);}}
                                min="1"
                                max="50"
                                className="w-full bg-zinc-900 border border-zinc-700 rounded-lg p-2.5 text-sm text-zinc-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                            />
                        </div>
                        <div className="space-y-2">
                            <label className="text-sm font-medium text-zinc-300 flex items-center gap-2">
                                <SettingsIcon className="w-4 h-4 text-amber-400" />
                                Target Device
                            </label>
                            <select
                                value={targetDevice}
                                onChange={lambda e: any -> None { setTargetDevice(e.target.value);}}
                                className="w-full bg-zinc-900 border border-zinc-700 rounded-lg p-2.5 text-sm text-zinc-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                            >
                                {available_devices.map(lambda d: dict -> Any {
                                    return <option key={d["value"]} value={d["value"]}>{d["label"]}</option>;
                                })}
                            </select>
                        </div>
                    </div>

                    <div className="space-y-2 pt-2">
                        <label className="flex items-center gap-2 cursor-pointer group">
                            <input
                                type="checkbox"
                                checked={useTemplate}
                                onChange={lambda e: any -> None { setUseTemplate(e.target.checked);}}
                                className="w-4 h-4 rounded border-zinc-700 bg-zinc-900 text-indigo-600 focus:ring-indigo-500/50"
                            />
                            <span className="text-sm text-zinc-300 group-hover:text-white transition-colors">
                                Use Baseline as Template
                            </span>
                        </label>
                        <p className="text-[10px] text-zinc-500 pl-6">
                            Inject PyTorch code as a reference template in the prompt
                        </p>
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
                        onClick={handleGenerate}
                        disabled={isWorking}
                        className={"px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-all shadow-lg hover:shadow-indigo-500/20 flex items-center gap-2 " + (
                            "opacity-50 cursor-not-allowed" if isWorking else ""
                        )}
                    >
                        {"Starting..." if isWorking else "Start Generation"}
                        <PlusIcon className={"w-4 h-4 " + ("opacity-0" if isWorking else "")} />
                    </button>
                </div>
            </div>
        </Modal>
    );
}
