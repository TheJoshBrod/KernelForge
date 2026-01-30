"""New Project Component"""

import from react { useEffect }
import from sonner { toast }
import from "@jac/runtime" { jacSpawn }
import from "react-router-dom" { useNavigate }

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

def:pub NewProject -> any {
    navigate = useNavigate();
    has code: str = "",
        formData: dict = {
            "projectName": "",
            "modelFile": None,
            "weightsFile": None,
            "validationZip": None
        },
        isSubmitting: bool = False;

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

        try {
            weights_b64 = "";
            if formData["weightsFile"] {
                weights_b64 = await fileToBase64(formData["weightsFile"]);
            }

            validation_b64 = "";
            validation_name = "";
            if formData["validationZip"] {
                validation_b64 = await fileToBase64(formData["validationZip"]);
                validation_name = formData["validationZip"].name;
            }

            # Call the walker passed from parent
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
                toast.success("Project created successfully");
                navigate("/project/" + result["name"]);
            } else {
                error_msg = result["error"] if result else "Failed to create project";
                toast.error(error_msg);
                isSubmitting = False;
            }
        } except Exception as e {
            console.error(e);
            toast.error("An error occurred while creating the project");
            isSubmitting = False;
        }
    }

    return
        <main
            className="flex h-screen bg-zinc-950 text-zinc-100 overflow-hidden relative"
        >
            <div className="absolute top-2 right-6">
                <a
                    onClick={lambda -> None { navigate("/settings");}}
                    className="flex items-center justify-center p-2 rounded-full hover:bg-zinc-800 transition-colors text-zinc-400 hover:text-white cursor-pointer"
                    aria-label="Settings"
                >
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    >
                        <path
                            d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.1a2 2 0 0 1-1-1.74v-.51a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"
                        />
                        <circle cx="12" cy="12" r="3" />
                    </svg>
                </a>
            </div>
            <div className="absolute top-2 right-16 z-10">
                <a
                    onClick={lambda -> None { navigate("/");}}
                    className="flex items-center justify-center p-2 rounded-full hover:bg-zinc-800 transition-colors text-zinc-400 hover:text-white bg-zinc-950/50 backdrop-blur-sm cursor-pointer"
                    aria-label="Go Back"
                >
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    >
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="15" y1="9" x2="9" y2="15"></line>
                        <line x1="9" y1="9" x2="15" y2="15"></line>
                    </svg>
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
                                    <svg
                                        xmlns="http://www.w3.org/2000/svg"
                                        width="24"
                                        height="24"
                                        viewBox="0 0 24 24"
                                        fill="none"
                                        stroke="currentColor"
                                        strokeWidth="2"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        className="text-zinc-400"
                                    >
                                        <path
                                            d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"
                                        />
                                        <polyline points="17 8 12 3 7 8" />
                                        <line x1="12" x2="12" y1="3" y2="15" />
                                    </svg>
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
                                className="w-full h-12 bg-zinc-900 border border-zinc-800 group-hover:border-zinc-700 rounded-lg px-4 flex items-center text-zinc-400 transition-all"
                            >
                                <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    width="16"
                                    height="16"
                                    viewBox="0 0 24 24"
                                    fill="none"
                                    stroke="currentColor"
                                    strokeWidth="2"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    className="mr-3"
                                >
                                    <path
                                        d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"
                                    />
                                </svg>
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
