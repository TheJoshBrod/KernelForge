"""New Project Component"""

import from react { useState, useEffect }
import from sonner { toast }
import from "react-simple-code-editor" { default as Editor }
import from "prismjs/components/prism-core" { highlight, languages }
import "prismjs/components/prism-clike";
import "prismjs/components/prism-python";
import "prismjs/themes/prism-tomorrow.css";
import "prismjs/themes/prism-tomorrow.css";
import "prismjs/themes/prism-tomorrow.css";
import from ..assets.SettingsIcon { SettingsIcon }
import from ..assets.BackIcon { BackIcon }
import from ..assets.UploadIcon { UploadIcon }
import from ..assets.FolderIcon { FolderIcon }

# Client-side helper for reading file as base64
def fileToBase64(file: any) -> any {
    return Reflect.construct(
        Promise,
        [
            lambda resolve: any , reject: any  -> None { reader = Reflect.construct(
                FileReader, []
            );reader.readAsDataURL(file);reader.onload = lambda -> None { resolve(
                reader.result.split(",")[1]
            );};reader.onerror = lambda error: any  -> None { reject(error);};}
        ]
    );
}

def:pub NewProject(onNavigate: any, createProjectWalker: any) -> any {
    has code: str = """# Paste your model code here

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)
""";
    has formData: dict = {
        "projectName": "",
        "weightsFile": None,
        "validationSet": None
    };
    has isSubmitting: bool = False;

    def handleInputChange(field: str, value: any) -> None {
        newFormData = {** formData};
        newFormData[field] = value;
        formData = newFormData;
    }

    # Handle Escape key to navigate back
    useEffect(
        lambda -> any { def handleKeyDown(event: any) -> None {
            if event.key == "Escape" {
                onNavigate("/");
            }
        } window.addEventListener("keydown", handleKeyDown);return lambda -> None { window.removeEventListener(
            "keydown", handleKeyDown
        );}; },
        [onNavigate]
    );

    async def handleCreate() -> None {
        if not formData["projectName"].trim() {
            toast.error("Project name is required");
            return;
        }

        isSubmitting = True;

        try {
            weights_b64 = "";
            if formData["weightsFile"] {
                weights_b64 = await fileToBase64(formData["weightsFile"]);
            }

            validationSetPath = "";
            if formData["validationSet"] and formData["validationSet"].length > 0 {
                firstFile = formData["validationSet"][0];
                parts = firstFile.webkitRelativePath.split("/");
                validationSetPath = parts[0] if parts.length > 0 else "";
            }

            # Call the walker passed from parent
            result = await createProjectWalker(
                formData["projectName"].trim(), code, weights_b64, validationSetPath
            );

            if result and result["success"] {
                toast.success("Project created successfully");
                onNavigate("/project/" + result["name"]);
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
                    onClick={lambda -> None { onNavigate("/settings");}}
                    className="flex items-center justify-center p-2 rounded-full hover:bg-zinc-800 transition-colors text-zinc-400 hover:text-white cursor-pointer"
                    aria-label="Settings"
                >
                    <SettingsIcon />
                </a>
            </div>
            <div className="absolute top-2 right-16 z-10">
                <a
                    onClick={lambda -> None { onNavigate("/");}}
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
                    <Editor
                        value={code}
                        onValueChange={lambda c: any  -> None { code = c;}}
                        highlight={lambda c: any  -> any { return highlight(
                            c, languages.python
                        ); }}
                        padding={24}
                        style={{
                            "fontFamily": '"Fira Code", "Fira Mono", monospace',
                            "fontSize": 14,
                            "minHeight": "100%",

                        }}
                        className="min-h-full"
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
                            Validation Set
                        </label>
                        <div className="relative group">
                            <input
                                type="file"
                                webkitdirectory=""
                                directory=""
                                onChange={lambda e: any  -> None { handleInputChange(
                                    "validationSet", e.target.files
                                );}}
                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                            />
                            <div
                                className="w-full h-12 bg-zinc-900 border border-zinc-800 group-hover:border-zinc-600 rounded-lg px-4 flex items-center text-zinc-400 transition-all"
                            >
                                <FolderIcon className="mr-3" />
                                <span className="truncate">
                                    {f"{formData['validationSet'].length} files selected"
                                    if formData["validationSet"]
                                    and formData["validationSet"].length > 0
                                    else "Select Directory..."}
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
