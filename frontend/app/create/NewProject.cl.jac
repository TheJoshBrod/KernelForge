"""New Project Component"""

import from react { useEffect, useState }
import from sonner { toast }
import from "react-router-dom" { useNavigate }
import from ...context.ConfigContext { useConfig }
import from ...components.ui.Icons { SettingsIcon, BackIcon, UploadIcon, FolderIcon }
import from ...components.editor.ForgePythonEditor { ForgePythonEditor }
import from "@jac-client/assets/KernelForge.svg" { default as loadingLogo }

# Client-side helper for reading file as base64
def fileToBase64(file: Any) -> Any {
    def executor(resolve: Any, reject: Any) -> None {
        reader = Reflect.construct(FileReader, []);
        reader.readAsDataURL(file);
        reader.onload = lambda -> None { resolve.call(
            None, reader.result.split(",")[1]
        );};
        reader.onerror = lambda -> None { reject.call(
            None, "Failed to read file as base64"
        );};
    }
    return Reflect.construct(Promise, [executor]);
}

def fileToText(file: Any) -> Any {
    def executor(resolve: Any, reject: Any) -> None {
        reader = Reflect.construct(FileReader, []);
        reader.readAsText(file);
        reader.onload = lambda -> None { resolve.call(None, reader.result);};
        reader.onerror = lambda -> None { reject.call(
            None, "Failed to read file as text"
        );};
    }
    return Reflect.construct(Promise, [executor]);
}

def:pub NewProject -> Any {
    navigate = useNavigate();
    config_ctx = useConfig();
    config = config_ctx.config;

    stageCycle = [
        "Heating up the coals",
        "Stoking the bellows",
        "Preparing the forge",
        "Tempering model steel",
        "Shaping kernels on the anvil",
        "Quenching fresh optimizations",
        "Cleaning the workspace"
    ];

    # Resume state if returning from settings
    persisted_state = window["__kf_new_project_state"] if "__kf_new_project_state" in window and window["__kf_new_project_state"] else {};
    initial_code = persisted_state["code"] if "code" in persisted_state else "";
    initial_project_name = persisted_state["projectName"] if "projectName" in persisted_state else "";

    code_state = useState(initial_code);
    code = code_state[0];
    setCode = code_state[1];

    form_data_state = useState({
        "projectName": initial_project_name,
        "modelFile": None,
        "weightsFile": None,
        "validationZip": None
    });
    formData = form_data_state[0];
    setFormData = form_data_state[1];

    is_submitting_state = useState(False);
    isSubmitting = is_submitting_state[0];
    setIsSubmitting = is_submitting_state[1];

    submit_stage_state = useState("");
    submitStage = submit_stage_state[0];
    setSubmitStage = submit_stage_state[1];

    stage_index_state = useState(0);
    stageIndex = stage_index_state[0];
    setStageIndex = stage_index_state[1];

    # Backend Selection Logic
    selected_backend_state = useState("");
    selectedBackend = selected_backend_state[0];
    setSelectedBackend = selected_backend_state[1];

    selected_gpu_info = config["selected_gpu_info"] if config and "selected_gpu_info" in config else None;
    compatible_backends = [];
    if selected_gpu_info {
        hint = selected_gpu_info["backend_hint"] if "backend_hint" in selected_gpu_info else "";
        if hint == "cuda" {
            compatible_backends = [{"label": "Triton", "value": "triton"}, {"label": "CUDA", "value": "cuda"}];
        } elif hint == "rocm" {
            compatible_backends = [{"label": "Triton", "value": "triton"}];
        } elif hint == "metal" {
            compatible_backends = [{"label": "Metal", "value": "metal"}];
        }
    } else {
        hwInfo = config_ctx.hwInfo if "hwInfo" in config_ctx else {};
        if hwInfo and "cuda_available" in hwInfo and hwInfo["cuda_available"] {
            compatible_backends = [{"label": "Triton", "value": "triton"}, {"label": "CUDA", "value": "cuda"}];
        } elif hwInfo and "mps_available" in hwInfo and hwInfo["mps_available"] {
            compatible_backends = [{"label": "Metal", "value": "metal"}];
        }
    }

    useEffect(
        lambda -> None {
            if compatible_backends.length > 0 {
                valid = False;
                for b in compatible_backends {
                    if b["value"] == selectedBackend {
                        valid = True;
                        break;
                    }
                }
                if not valid {
                    setSelectedBackend(compatible_backends[0]["value"]);
                }
            } else {
                setSelectedBackend("");
            }
        },
        [selected_gpu_info, config_ctx.hwInfo, selectedBackend]
    );

    useEffect(
        lambda -> None {
            window["__kf_new_project_state"] = {
                "projectName": formData["projectName"],
                "code": code
            };
        },
        [formData["projectName"], code]
    );

    def handleInputChange(field: str, value: Any) -> None {
        newFormData = {
            "projectName": formData["projectName"],
            "modelFile": formData["modelFile"],
            "weightsFile": formData["weightsFile"],
            "validationZip": formData["validationZip"]
        };
        newFormData[field] = value;
        setFormData(newFormData);
    }

    def previewModelFile(file: Any) -> None {
        if not file {
            return;
        }
        handleInputChange("modelFile", file);
        reader = Reflect.construct(FileReader, []);
        reader.onload = lambda -> None { text_value = reader.result
        if reader.result
        else "";if text_value and text_value.trim() {
            setCode(text_value);
        } else {
            toast.error("Selected model file is empty");
        } };
        reader.onerror = lambda -> None { toast.error(
            "Failed to read model file for preview"
        );};
        reader.readAsText(file);
    }

    def handleKeyDown(event: Any) -> None {
        if event.key == "Escape" {
            navigate("/");
        }
    }

    useEffect(
        lambda -> Any { window.addEventListener("keydown", handleKeyDown);return lambda -> None { window.removeEventListener(
            "keydown", handleKeyDown
        );}; },
        []
    );

    useEffect(
        lambda -> Any { if not isSubmitting {
            setStageIndex(0);
            return;
        }setStageIndex(Math.floor(Date.now() / 1300) % stageCycle.length);timer = setInterval(
            lambda -> None { setStageIndex(Math.floor(Date.now() / 1300) % stageCycle.length);},
            650
        );return lambda -> None { clearInterval(timer);}; },
        [isSubmitting]
    );

    async def handleCreate -> None {
        if not formData["projectName"].trim() {
            toast.error("Project name is required");
            return;
        }
        if (not code or not code.trim()) and not formData["modelFile"] {
            toast.error("Model code or model file is required");
            return;
        }

        setIsSubmitting(True);
        setSubmitStage("Preparing project files");

        try {
            model_code = code;
            model_file_b64 = "";
            if formData["modelFile"] {
                setSubmitStage("Encoding model file");
                model_file_b64 = await fileToBase64(formData["modelFile"]);
                if not model_code or not model_code.trim() {
                    try {
                        model_code = await fileToText(formData["modelFile"]);
                        setCode(model_code);
                    } except Exception as e {
                        console.error(e);
                    }
                }
            }
            if (not model_code or not model_code.trim()) and not model_file_b64 {
                toast.error("Failed to load model code from file");
                setIsSubmitting(False);
                setSubmitStage("");
                return;
            }

            weights_b64 = "";
            if formData["weightsFile"] {
                setSubmitStage("Encoding weights");
                weights_b64 = await fileToBase64(formData["weightsFile"]);
            }

            validation_b64 = "";
            validation_name = "";
            if formData["validationZip"] {
                setSubmitStage("Encoding validation set");
                validation_b64 = await fileToBase64(formData["validationZip"]);
                validation_name = formData["validationZip"].name;
            }

            setSubmitStage("Creating project");
            resp = await __jacSpawn(
                "CreateProject",
                "",
                {
                    "projectName": formData["projectName"].trim(),
                    "selectedGpuInfo": config["selected_gpu_info"] if config and "selected_gpu_info" in config else None,
                    "selectedBackend": selectedBackend,
                    "code": model_code,
                    "modelFileB64": model_file_b64,
                    "modelFileName": (
                        formData["modelFile"].name
                        if formData["modelFile"] and "name" in formData["modelFile"]
                        else "model.py"
                    ),
                    "weightsFileB64": weights_b64,
                    "weightsFileName": (
                        formData["weightsFile"].name
                        if formData["weightsFile"]
                        and "name" in formData["weightsFile"]
                        else "weights.pt"
                    ),
                    "validationSetPath": "",
                    "validationZipB64": validation_b64,
                    "validationZipName": validation_name
                }
            );
            result = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else None;

            if result and result["success"] {
                setSubmitStage("Finalizing");
                toast.success("Project created successfully");
                window["__kf_new_project_state"] = None;
                navigate("/project/" + result["name"], {"replace": True});
            } else {
                error_msg = result["error"] if result else "Failed to create project";
                toast.error(error_msg);
                setIsSubmitting(False);
                setSubmitStage("");
            }
        } except Exception as e {
            console.error(e);
            toast.error("An error occurred while creating the project");
            setIsSubmitting(False);
            setSubmitStage("");
        }
    }

    loadingOverlay = None;
    if isSubmitting {
        loadingOverlay = <div className="project-loading">
            <div className="project-loading-card">
                <img
                    src={loadingLogo}
                    alt="Kernal Forge Logo"
                    className="project-loading-logo"
                />
                <div className="project-loading-title">
                    Creating Project
                </div>
                <div className="project-loading-stage">
                    {stageCycle[stageIndex]}
                </div>
                <div className="project-loading-bar">
                    <span className="project-loading-bar-fill"></span>
                </div>
                <div className="project-loading-foot">
                    Status:{" "+ submitStage or " Working on it..."} | Large weights or validation zips may take a moment.
                </div>
            </div>
        </div>;
    }

    return
        <main className="app-shell">
            {loadingOverlay}
            <div className="app-backdrop app-backdrop--forge"></div>
            <div className="app-backdrop-embers">
                <span className="ember-particle"></span>
                <span className="ember-particle"></span>
                <span className="ember-particle"></span>
                <span className="ember-particle"></span>
                <span className="ember-particle"></span>
                <span className="ember-particle"></span>
            </div>
            <div className="app-content-layer h-screen w-full p-4 md:p-6">
                <div
                    className="h-full w-full border border-[var(--border)] bg-[var(--surface-0)] clip-anvil shadow-[0_22px_48px_rgba(0,0,0,0.45)] overflow-hidden"
                >
                    <div
                        className="h-14 flex items-center justify-between px-6 border-b border-[var(--border)] bg-[var(--surface-1)] relative"
                    >
                        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/15 to-transparent"></div>
                        <div>
                            <h1 className="text-sm font-bold uppercase tracking-[0.18em] text-[var(--text-strong)]">
                                Create Project
                            </h1>
                            <p className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">
                                Kernel Forge Onboarding
                            </p>
                        </div>
                        <div className="flex items-center gap-2">
                            <a
                                onClick={lambda -> None { navigate("/");}}
                                className="flex items-center justify-center p-2 rounded-full border border-[var(--border)] hover:border-[var(--border-strong)] bg-[var(--surface-0)] transition-colors text-zinc-400 hover:text-white cursor-pointer"
                                aria-label="Go Back"
                            >
                                <BackIcon/>
                            </a>
                            <a
                                onClick={lambda -> None { navigate(
                                    "/settings", {"state": {"return_to": "/new-project"}}
                                );}}
                                className="flex items-center justify-center p-2 rounded-full border border-[var(--border)] hover:border-[var(--border-strong)] bg-[var(--surface-0)] transition-colors text-zinc-400 hover:text-white cursor-pointer"
                                aria-label="Settings"
                            >
                                <SettingsIcon/>
                            </a>
                        </div>
                    </div>
                    <div className="h-[calc(100%-3.5rem)] flex">
                        <div className="w-1/2 min-w-0 flex flex-col border-r border-[var(--border)]">
                            <div
                                className="h-14 flex items-center px-6 border-b border-[var(--border)] bg-[var(--surface-0)] relative"
                            >
                                <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>
                                <h2
                                    className="text-sm font-bold uppercase tracking-wider text-zinc-400"
                                >
                                    Model Code
                                </h2>
                            </div>
                            <div className="flex-1 relative">
                                <ForgePythonEditor
                                    value={code}
                                    onChange={lambda next_code: Any  -> None { setCode(str(
                                        next_code
                                    )
                                    if next_code
                                    else "");}}
                                    readOnly={isSubmitting}
                                    placeholder="# Paste your model code here or upload model.py"
                                />
                            </div>
                        </div>
                        <div className="w-1/2 min-w-0 flex flex-col bg-[var(--surface-0)]">
                            <div
                                className="h-14 flex items-center justify-between px-6 border-b border-[var(--border)] bg-[var(--surface-0)] relative"
                            >
                                <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>
                                <h2
                                    className="text-sm font-bold uppercase tracking-wider text-zinc-400"
                                >
                                    Project Details
                                </h2>
                            </div>
                            <div className="flex-1 overflow-y-auto">
                                <div className="p-8 space-y-8 max-w-xl mx-auto w-full">
                                    <div className="space-y-2">
                                        <label
                                            className="text-xs font-semibold text-zinc-500 uppercase tracking-wider"
                                        >
                                            Project Name
                                        </label>
                                        <input
                                            type="text"
                                            value={formData["projectName"]}
                                            onChange={lambda e: any  -> None { handleInputChange(
                                                "projectName", e.target.value
                                            );}}
                                            placeholder="e.g. ResNet-50 Fine-tune"
                                            className="w-full h-12 bg-[var(--surface-2)] border border-[var(--border)] hover:border-[var(--border-strong)] rounded-lg px-4 text-zinc-200 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-orange-400/30 transition-all"
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <div className="flex justify-between items-center">
                                            <label
                                                className="text-xs font-semibold text-zinc-500 uppercase tracking-wider"
                                            >
                                                Target & Backend Options
                                            </label>
                                            <div className="flex items-center gap-2">
                                                <span className="text-xs text-zinc-400">
                                                    Target: <span className="text-amber-400 font-medium">{selected_gpu_info["name"] if selected_gpu_info and "name" in selected_gpu_info else "Local GPU"}</span>
                                                </span>
                                                <a 
                                                    onClick={lambda -> None { navigate("/settings/gpu", {"state": {"return_to": "/new-project"}}); }}
                                                    className="text-[10px] text-zinc-500 hover:text-indigo-400 cursor-pointer underline flex items-center"
                                                >
                                                    Change GPU
                                                </a>
                                            </div>
                                        </div>
                                        {<div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
                                            <p className="text-sm text-red-400">
                                                No compatible backend found for this GPU. Supported backends: CUDA, Triton, Metal.
                                            </p>
                                        </div>
                                        if compatible_backends.length == 0
                                        else <div className="relative">
                                            <select
                                                value={selectedBackend}
                                                onChange={lambda e: any -> None { setSelectedBackend(e.target.value); }}
                                                className="w-full h-12 bg-[#1a1c21] border border-[var(--border)] hover:border-[var(--border-strong)] rounded-lg px-4 text-[13px] text-zinc-300 font-medium focus:outline-none focus:ring-2 focus:ring-orange-400/30 appearance-none shadow-inner"
                                            >
                                                {compatible_backends.map(
                                                    lambda item: dict -> Any { return
                                                        <option key={item["value"]} value={item["value"]} className="bg-[#1a1c21] text-zinc-300 py-2">
                                                            {item["label"]}
                                                        </option>;
                                                    }
                                                )}
                                            </select>
                                            <div className="absolute inset-y-0 right-0 flex items-center px-4 pointer-events-none">
                                                <svg className="w-4 h-4 text-zinc-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
                                            </div>
                                        </div>}
                                    </div>
                                    <div className="space-y-2">
                                        <label
                                            className="text-xs font-semibold text-zinc-500 uppercase tracking-wider"
                                        >
                                            Model Code (.py)
                                        </label>
                                        <div className="relative group">
                                            <input
                                                type="file"
                                                accept=".py,.txt"
                                                onChange={lambda e: any  -> None { previewModelFile(
                                                    e.target.files[0]
                                                );}}
                                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                                            />
                                            <div
                                                className="w-full h-20 bg-[var(--surface-2)] border border-[var(--border)] group-hover:border-[var(--border-strong)] clip-anvil flex flex-col items-center justify-center transition-colors px-4"
                                            >
                                                <p className="text-sm text-zinc-300 font-medium truncate w-full text-center">
                                                    {formData["modelFile"].name
                                                    if formData["modelFile"]
                                                    else "Click to select model.py"}
                                                </p>
                                                <p className="text-xs text-zinc-500 mt-1">
                                                    Uploading will replace the editor content
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="space-y-2">
                                        <label
                                            className="text-xs font-semibold text-zinc-500 uppercase tracking-wider"
                                        >
                                            Upload Weights (.pt or .pth)
                                        </label>
                                        <div className="relative group">
                                            <input
                                                type="file"
                                                accept=".pt,.pth"
                                                onChange={lambda e: any  -> None { handleInputChange(
                                                    "weightsFile", e.target.files[0]
                                                );}}
                                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                                            />
                                            <div
                                                className="w-full h-32 bg-[var(--surface-2)] border border-[var(--border)] group-hover:border-[var(--border-strong)] clip-anvil flex flex-col items-center justify-center transition-colors px-4"
                                            >
                                                <div className="p-3 rounded-full bg-[var(--surface-1)] mb-2">
                                                    <UploadIcon className="text-zinc-400"/>
                                                </div>
                                                <p className="text-sm text-zinc-300 font-medium truncate w-full text-center">
                                                    {formData["weightsFile"].name
                                                    if formData["weightsFile"]
                                                    else "Click to select or drag file"}
                                                </p>
                                                <p className="text-xs text-zinc-500 mt-1">
                                                    PyTorch checkpoint file
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="space-y-2">
                                        <label
                                            className="text-xs font-semibold text-zinc-500 uppercase tracking-wider"
                                        >
                                            Validation Set (.zip)
                                        </label>
                                        <div className="relative group">
                                            <input
                                                type="file"
                                                accept=".zip"
                                                onChange={lambda e: any  -> None { handleInputChange(
                                                    "validationZip", e.target.files[0]
                                                );}}
                                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                                            />
                                            <div
                                                className="w-full h-12 bg-[var(--surface-2)] border border-[var(--border)] group-hover:border-[var(--border-strong)] clip-anvil px-4 flex items-center text-zinc-300 transition-colors"
                                            >
                                                <FolderIcon className="mr-3 text-zinc-400"/>
                                                <span className="truncate">
                                                    {formData["validationZip"].name
                                                    if formData["validationZip"]
                                                    else "Select .zip file..."}
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="pt-6">
                                        <button
                                            onClick={lambda e: any  -> None { handleCreate();}}
                                            disabled={isSubmitting or compatible_backends.length == 0}
                                            className="w-full h-12 bg-[var(--accent-orange)] hover:bg-[var(--molten-gold-dark)] text-zinc-950 font-bold rounded-lg transition-all shadow-[0_10px_24px_rgba(225,112,85,0.28)] hover:shadow-[0_12px_28px_rgba(232,168,56,0.26)] disabled:opacity-50 disabled:cursor-not-allowed"
                                        >
                                            {"Creating..." if isSubmitting else "Create Project"}
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </main>;
}
