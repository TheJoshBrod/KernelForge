"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import Editor from "react-simple-code-editor";
import { highlight, languages } from "prismjs/components/prism-core";
import "prismjs/components/prism-clike";
import "prismjs/components/prism-python";
import "prismjs/themes/prism-tomorrow.css"; // Dark theme for code

import { toast } from "sonner";

export default function NewProject() {
    const router = useRouter();
    const [code, setCode] = useState(`# Paste your model code here\n\nimport torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n        pass\n`);
    const [formData, setFormData] = useState({
        projectName: "",
        weightsFile: null,
        validationSet: null,
    });
    const [isSubmitting, setIsSubmitting] = useState(false);

    const handleInputChange = (field, value) => {
        setFormData((prev) => ({ ...prev, [field]: value }));
    };

    useEffect(() => {
        const handleKeyDown = (event) => {
            if (event.key === "Escape") {
                router.push("/");
            }
        };

        window.addEventListener("keydown", handleKeyDown);
        return () => window.removeEventListener("keydown", handleKeyDown);
    }, [router]);

    const handleCreate = async () => {
        if (!formData.projectName.trim()) {
            toast.error("Project name is required");
            return;
        }

        setIsSubmitting(true);
        const data = new FormData();
        data.append("projectName", formData.projectName.trim());
        data.append("code", code);

        if (formData.weightsFile) {
            data.append("weightsFile", formData.weightsFile);
        }

        // Handle validation set path
        // Note: For webkitdirectory, files array contains relative paths, but we can't get absolute path in browser due to security.
        // We will store the path if accessible, or just the fact that it was selected.
        // For local development on same machine, user might expect the full path, but web browsers don't give it.
        // We'll proceed with storing what we can or a placeholder if security prevents full path access.
        // Ideally user would input a path string for server-side processing or we accept uploaded files.
        // Given prompt requirements "save location of", we can't reliably get full absolute path from browser input type=file.
        // We will mock this or use the relative path property if available for the sake of the requirement, 
        // acknowledging browser limitations.
        // For the purpose of this task (which assumes a local context or specific environment), we'll check if we can get a path.
        // If not, we'll iterate files to find the common directory name.
        if (formData.validationSet && formData.validationSet.length > 0) {
            // Best effort to get a directory name/path
            const firstFile = formData.validationSet[0];
            // webkitRelativePath usually looks like "Folder/File.txt"
            const dirName = firstFile.webkitRelativePath.split('/')[0];
            data.append("validationSetPath", dirName);
        }

        try {
            const res = await fetch("/api/projects", {
                method: "POST",
                body: data,
            });

            if (res.status === 409) {
                toast.error("A project with this name already exists");
                setIsSubmitting(false);
                return;
            }

            if (!res.ok) {
                throw new Error("Failed to create project");
            }

            const json = await res.json();
            toast.success("Project created successfully");
            router.push(`/project/${json.name}`);

        } catch (error) {
            console.error(error);
            toast.error("An error occurred while creating the project");
            setIsSubmitting(false);
        }
    };

    return (
        <main className="flex h-screen bg-zinc-950 text-zinc-100 overflow-hidden relative">
            {/* Go Back Button - Top Right */}
            <div className="absolute top-2 right-6">
                <Link
                href="/settings"
                className="flex items-center justify-center p-2 rounded-full hover:bg-zinc-800 transition-colors text-zinc-400 hover:text-white"
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
                    <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.1a2 2 0 0 1-1-1.74v-.51a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
                    <circle cx="12" cy="12" r="3" />
                </svg>
                </Link>
            </div>
            <div className="absolute top-2 right-16 z-10">
                <Link
                    href="/"
                    className="flex items-center justify-center p-2 rounded-full hover:bg-zinc-800 transition-colors text-zinc-400 hover:text-white bg-zinc-950/50 backdrop-blur-sm"
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
                </Link>
            </div>
            {/* Left Panel - Code Editor */}
            <div className="w-1/2 flex flex-col border-r border-zinc-900">
                <div className="h-14 flex items-center px-6 border-b border-zinc-900 bg-zinc-950">
                    <h2 className="text-sm font-bold uppercase tracking-wider text-zinc-400">Model Code</h2>
                </div>
                <div className="flex-1 overflow-auto bg-[#1d1f21] relative custom-scrollbar">
                    <Editor
                        value={code}
                        onValueChange={(code) => setCode(code)}
                        highlight={(code) => highlight(code, languages.python)}
                        padding={24}
                        style={{
                            fontFamily: '"Fira Code", "Fira Mono", monospace',
                            fontSize: 14,
                            minHeight: "100%",
                        }}
                        className="min-h-full"
                    />
                </div>
            </div>

            {/* Right Panel - Configuration */}
            <div className="w-1/2 flex flex-col bg-zinc-950">
                <div className="h-14 flex items-center justify-between px-6 border-b border-zinc-900">
                    <h2 className="text-sm font-bold uppercase tracking-wider text-zinc-400">Project Details</h2>
                </div>

                <div className="p-8 space-y-8 max-w-xl mx-auto w-full mt-10">
                    {/* Project Name */}
                    <div className="space-y-2">
                        <label className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">
                            Project Name
                        </label>
                        <input
                            type="text"
                            value={formData.projectName}
                            onChange={(e) => handleInputChange("projectName", e.target.value)}
                            placeholder="e.g. ResNet-50 Fine-tune"
                            className="w-full h-12 bg-zinc-900 border border-zinc-800 hover:border-zinc-700 rounded-lg px-4 text-zinc-200 placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all"
                        />
                    </div>

                    {/* Upload Weights */}
                    <div className="space-y-2">
                        <label className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">
                            Upload Weights (.pt)
                        </label>
                        <div className="relative group">
                            <input
                                type="file"
                                accept=".pt"
                                onChange={(e) => handleInputChange("weightsFile", e.target.files[0])}
                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                            />
                            <div className="w-full h-32 bg-zinc-900/50 border-2 border-dashed border-zinc-800 group-hover:border-zinc-600 rounded-xl flex flex-col items-center justify-center transition-all">
                                <div className="p-3 rounded-full bg-zinc-900 mb-2">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-zinc-400"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" x2="12" y1="3" y2="15" /></svg>
                                </div>
                                <p className="text-sm text-zinc-400 font-medium">
                                    {formData.weightsFile ? formData.weightsFile.name : "Click to select or drag file"}
                                </p>
                                <p className="text-xs text-zinc-600 mt-1">PyTorch weights only</p>
                            </div>
                        </div>
                    </div>

                    {/* Validation Set */}
                    <div className="space-y-2">
                        <label className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">
                            Validation Set
                        </label>
                        <div className="relative group">
                            <input
                                type="file"
                                {...{ webkitdirectory: "", directory: "" }}
                                onChange={(e) => handleInputChange("validationSet", e.target.files)}
                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                            />
                            <div className="w-full h-12 bg-zinc-900 border border-zinc-800 group-hover:border-zinc-700 rounded-lg px-4 flex items-center text-zinc-400 transition-all">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-3"><path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z" /></svg>
                                <span className="truncate">
                                    {formData.validationSet && formData.validationSet.length > 0
                                        ? `${formData.validationSet.length} files selected`
                                        : "Select Directory..."}
                                </span>
                            </div>
                        </div>
                    </div>

                    <div className="pt-6">
                        <button
                            onClick={handleCreate}
                            disabled={isSubmitting}
                            className="w-full h-12 bg-zinc-100 hover:bg-white text-zinc-950 font-bold rounded-lg transition-all shadow-lg hover:shadow-zinc-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {isSubmitting ? "Creating..." : "Create Project"}
                        </button>
                    </div>
                </div>
            </div>
        </main>
    );
}
