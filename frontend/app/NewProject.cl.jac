"""New Project Component"""

import from react { useEffect }
import from sonner { toast }
import from "@jac/runtime" { jacSpawn }
import from "react-router-dom" { useNavigate }
import from "../assets/SettingsIcon.cl.jac" { SettingsIcon }
import from "../assets/BackIcon.cl.jac" { BackIcon }
import from "../assets/UploadIcon.cl.jac" { UploadIcon }
import from "../assets/FolderIcon.cl.jac" { FolderIcon }

def LoadingGhostIcon(className: str = "") -> Any {
    return
        <svg viewBox="0 0 120 120" className={className} fill="currentColor">
            <path
                d="M 60 10 C 30 10 10 40 10 70 L 10 100 C 10 110 30 110 30 100 C 30 110 50 110 50 100 C 50 110 70 110 70 100 C 70 110 90 110 90 100 C 90 110 110 110 110 100 L 110 70 C 110 40 90 10 60 10 Z"
            />
            <circle cx="45" cy="50" r="5" fill="#0b0f14" />
            <circle cx="75" cy="50" r="5" fill="#0b0f14" />
            <ellipse cx="60" cy="70" rx="4" ry="3" fill="#0b0f14" opacity="0.2" />
        </svg>;
}

# Client-side helper for reading file as base64
def fileToBase64(file: any) -> any {
    def executor(resolve: any, reject: any) -> None {
        reader = Reflect.construct(FileReader, []);
        reader.readAsDataURL(file);
        reader.onload = lambda -> None { resolve(reader.result.split(",")[1]);};
        reader.onerror = lambda -> None { return; };
    }
    return Reflect.construct(Promise, [executor]);
}

def fileToText(file: any) -> any {
    def executor(resolve: any, reject: any) -> None {
        reader = Reflect.construct(FileReader, []);
        reader.readAsText(file);
        reader.onload = lambda -> None { resolve(reader.result);};
        reader.onerror = lambda -> None { return; };
    }
    return Reflect.construct(Promise, [executor]);
}

def:pub NewProject {
    navigate = useNavigate();
    has code: str = "",
        formData: dict = {
            "projectName": "",
            "modelFile": None,
            "weightsFile": None,
            "validationZip": None
        },
        isSubmitting: bool = False,
        submitStage: str = "";

    def handleInputChange(field: str, value: any) -> None {
        newFormData = {
            "projectName": formData["projectName"],
            "modelFile": formData["modelFile"],
            "weightsFile": formData["weightsFile"],
            "validationZip": formData["validationZip"]
        };
        newFormData[field] = value;
        formData = newFormData;
    }

    async def handleModelFile(file: any) -> None {
        if not file {
            return;
        }
        try {
            text = await fileToText(file);
            code = text;
            handleInputChange("modelFile", file);
        } except Exception as e {
            console.error(e);
            toast.error("Failed to read model file");
        }
    }

    # Handle Escape key to navigate back
    def handleKeyDown(event: any) -> None {
        if event.key == "Escape" {
            navigate("/");
        }
    }

    useEffect(
        lambda -> None { window.addEventListener("keydown", handleKeyDown);return lambda -> None { window.removeEventListener(
            "keydown", handleKeyDown
        );}; },
        []
    );

    async def handleCreate -> None {
        if not formData["projectName"].trim() {
            toast.error("Project name is required");
            return;
        }
        if not code.trim() {
            toast.error("Model code is required");
            return;
        }

        isSubmitting = True;
        submitStage = "Preparing project files";

        try {
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

            # Call the walker passed from parent
            submitStage = "Creating project";
            resp = await jacSpawn(
                "CreateProject",
                "",
                {
                    "projectName": formData["projectName"].trim(),
                    "code": code,
                    "weightsFileB64": weights_b64,
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
                navigate("/project/" + result["name"]);
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
                <LoadingGhostIcon className="project-loading-ghost" />
                <div className="project-loading-title">
                    Creating Project
                </div>
                <div className="project-loading-stage">
                    {submitStage or "Working on it..."}
                </div>
                <div className="project-loading-bar">
                    <span className="project-loading-bar-fill"></span>
                </div>
                <div className="project-loading-foot">
                    Large weights or validation zips may take a moment.
                </div>
            </div>
        </div>;
    }

    return
        <main
            className="flex h-screen bg-zinc-950 text-zinc-100 overflow-hidden relative"
        >
            {loadingOverlay}
            <div className="absolute top-2 right-6">
                <a
                    onClick={lambda -> None { navigate("/settings");}}
                    className="flex items-center justify-center p-2 rounded-full hover:bg-zinc-800 transition-colors text-zinc-400 hover:text-white cursor-pointer"
                    aria-label="Settings"
                >
                    <SettingsIcon />
                </a>
            </div>
            <div className="absolute top-2 right-16 z-10">
                <a
                    onClick={lambda -> None { navigate("/");}}
                    className="flex items-center justify-center p-2 rounded-full hover:bg-zinc-800 transition-colors text-zinc-400 hover:text-white bg-zinc-950/50 backdrop-blur-sm cursor-pointer"
                    aria-label="Go Back"
                >
                    <BackIcon />
                </a>
            </div>
            <div className="w-1/2 flex flex-col border-r border-zinc-900">
                <div
                    className="h-14 flex items-center px-6 border-b border-zinc-900 bg-zinc-950"
                >
                    <h2
                        className="text-sm font-bold uppercase tracking-wider text-zinc-400"
                    >
                        Model Code
                    </h2>
                </div>
                <div
                    className="flex-1 overflow-auto bg-[#1d1f21] relative custom-scrollbar"
                >
                    <textarea
                        value={code}
                        onChange={lambda e: any  -> None { code = e.target.value;}}
                        placeholder="# Paste your model code here or upload model.py"
                        className="min-h-full w-full resize-none bg-[#1d1f21] text-zinc-100 font-mono text-sm p-6 focus:outline-none placeholder:text-zinc-500"
                    />
                </div>
            </div>
            <div className="w-1/2 flex flex-col bg-zinc-950">
                <div
                    className="h-14 flex items-center justify-between px-6 border-b border-zinc-900"
                >
                    <h2
                        className="text-sm font-bold uppercase tracking-wider text-zinc-400"
                    >
                        Project Details
                    </h2>
                </div>
                <div className="p-8 space-y-8 max-w-xl mx-auto w-full mt-10">
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
                            className="w-full h-12 bg-zinc-900 border border-zinc-800 hover:border-zinc-700 rounded-lg px-4 text-zinc-200 placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all"
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
                                onChange={lambda e: any  -> None { handleModelFile(
                                    e.target.files[0]
                                );}}
                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                            />
                            <div
                                className="w-full h-20 bg-zinc-900/50 border-2 border-dashed border-zinc-800 group-hover:border-zinc-600 rounded-xl flex flex-col items-center justify-center transition-all"
                            >
                                <p className="text-sm text-zinc-400 font-medium">
                                    {formData["modelFile"].name
                                    if formData["modelFile"]
                                    else "Click to select model.py"}
                                </p>
                                <p className="text-xs text-zinc-600 mt-1">
                                    Uploading will replace the editor content
                                </p>
                            </div>
                        </div>
                    </div>
                    <div className="space-y-2">
                        <label
                            className="text-xs font-semibold text-zinc-500 uppercase tracking-wider"
                        >
                            Upload Weights (.pt)
                        </label>
                        <div className="relative group">
                            <input
                                type="file"
                                accept=".pt"
                                onChange={lambda e: any  -> None { handleInputChange(
                                    "weightsFile", e.target.files[0]
                                );}}
                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                            />
                            <div
                                className="w-full h-32 bg-zinc-900/50 border-2 border-dashed border-zinc-800 group-hover:border-zinc-600 rounded-xl flex flex-col items-center justify-center transition-all"
                            >
                                <div className="p-3 rounded-full bg-zinc-900 mb-2">
                                    <UploadIcon className="text-zinc-400" />
                                </div>
                                <p className="text-sm text-zinc-400 font-medium">
                                    {formData["weightsFile"].name
                                    if formData["weightsFile"]
                                    else "Click to select or drag file"}
                                </p>
                                <p className="text-xs text-zinc-600 mt-1">
                                    PyTorch weights only
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
                                className="w-full h-12 bg-zinc-900 border border-zinc-800 group-hover:border-zinc-600 rounded-lg px-4 flex items-center text-zinc-400 transition-all"
                            >
                                <FolderIcon className="mr-3" />
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
                            className="w-full h-12 bg-zinc-100 hover:bg-white text-zinc-950 font-bold rounded-lg transition-all shadow-lg hover:shadow-zinc-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {"Creating..." if isSubmitting else "Create Project"}
                        </button>
                    </div>
                </div>
            </div>
        </main>;
}
