"""New Project Component"""

import from react { useEffect }
import from sonner { toast }
import from "react-router-dom" { useNavigate }
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
    has code: str = "",
        formData: dict = {
            "projectName": "",
            "modelFile": None,
            "weightsFile": None,
            "validationZip": None
        },
        isSubmitting: bool = False,
        submitStage: str = "",
        stageIndex: int = 0,
        stageCycle: list = [
            "Heating up the coals",
            "Stoking the bellows",
            "Preparing the forge",
            "Tempering model steel",
            "Shaping kernels on the anvil",
            "Quenching fresh optimizations",
            "Cleaning the workspace"
        ];

    def handleInputChange(field: str, value: Any) -> None {
        newFormData = {
            "projectName": formData["projectName"],
            "modelFile": formData["modelFile"],
            "weightsFile": formData["weightsFile"],
            "validationZip": formData["validationZip"]
        };
        newFormData[field] = value;
        formData = newFormData;
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
            code = text_value;
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
            stageIndex = 0;
            return;
        }stageIndex = Math.floor(Date.now() / 1300) % stageCycle.length;timer = setInterval(
            lambda -> None { stageIndex = Math.floor(Date.now() / 1300) % stageCycle.length;},
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

        isSubmitting = True;
        submitStage = "Preparing project files";

        try {
            model_code = code;
            model_file_b64 = "";
            if formData["modelFile"] {
                submitStage = "Encoding model file";
                model_file_b64 = await fileToBase64(formData["modelFile"]);
                if not model_code or not model_code.trim() {
                    try {
                        model_code = await fileToText(formData["modelFile"]);
                        code = model_code;
                    } except Exception as e {
                        console.error(e);
                    }
                }
            }
            if (not model_code or not model_code.trim()) and not model_file_b64 {
                toast.error("Failed to load model code from file");
                isSubmitting = False;
                submitStage = "";
                return;
            }

            weights_b64 = "";
            if formData["weightsFile"] {
                submitStage = "Encoding weights";
                weights_b64 = await fileToBase64(formData["weightsFile"]);
            }

            validation_b64 = "";
            validation_name = "";
            if formData["validationZip"] {
                submitStage = "Encoding validation set";
                validation_b64 = await fileToBase64(formData["validationZip"]);
                validation_name = formData["validationZip"].name;
            }

            submitStage = "Creating project";
            resp = await __jacSpawn(
                "CreateProject",
                "",
                {
                    "projectName": formData["projectName"].trim(),
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
                submitStage = "Finalizing";
                toast.success("Project created successfully");
                navigate("/project/" + result["name"], {"replace": True});
            } else {
                error_msg = result["error"] if result else "Failed to create project";
                toast.error(error_msg);
                isSubmitting = False;
                submitStage = "";
            }
        } except Exception as e {
            console.error(e);
            toast.error("An error occurred while creating the project");
            isSubmitting = False;
            submitStage = "";
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
                    Status:{submitStage or "Working on it..."} | Large weights or validation zips may take a moment.
                </div>
            </div>
        </div>;
    }

    return
        <main className="app-shell">
            {loadingOverlay}
            <div className="app-backdrop app-backdrop--natural"></div>
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
                                    onChange={lambda next_code: Any  -> None { code = str(
                                        next_code
                                    )
                                    if next_code
                                    else "";}}
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
                                            disabled={isSubmitting}
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
