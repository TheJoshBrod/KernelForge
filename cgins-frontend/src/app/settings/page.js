"use client";

import { useRouter } from "next/navigation";
import { useConfig } from "../../context/ConfigContext";
import { useState, useEffect } from "react";

export default function Settings() {
    const router = useRouter();
    const { config, setConfig, saveConfig, loading } = useConfig();
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        const handleKeyDown = (event) => {
            if (event.key === "Escape") {
                router.push("/");
            }
        };

        window.addEventListener("keydown", handleKeyDown);
        return () => window.removeEventListener("keydown", handleKeyDown);
    }, [router]);

    // Directly update global state for immediate feedback/binding
    const handleChange = (field, value) => {
        setConfig((prev) => ({
            ...prev,
            llm_info: {
                ...prev.llm_info,
                [field]: value,
            },
        }));
    };

    const handleSave = async () => {
        setSaving(true);
        // Save to config.json via API
        await saveConfig(config);
        setSaving(false);
        router.push("/");
    };

    const handleReturn = () => {
        router.push("/");
    };

    if (loading) {
        return (
            <div className="flex min-h-screen items-center justify-center bg-zinc-950 text-zinc-100">
                <p>Loading settings...</p>
            </div>
        );
    }

    const { llm_info } = config;

    return (
        <main className="flex min-h-screen flex-col items-center justify-center p-6 bg-zinc-900 text-zinc-100 selection:bg-zinc-700">
            <div className="w-full max-w-md space-y-8 bg-zinc-950 p-8 rounded-xl border border-zinc-800 shadow-2xl">
                <div className="text-center space-y-2">
                    <h1 className="text-3xl font-bold tracking-tight text-white">Settings</h1>
                    <p className="text-zinc-500 text-sm">Configure your Model Provider</p>
                </div>

                <div className="space-y-6">
                    {/* Provider Dropdown */}
                    <div className="space-y-2">
                        <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                            Model Provider
                        </label>
                        <div className="relative">
                            <select
                                value={llm_info.provider}
                                onChange={(e) => handleChange("provider", e.target.value)}
                                className="w-full h-12 appearance-none bg-zinc-900 border border-zinc-800 hover:border-zinc-700 rounded-lg px-4 text-zinc-200 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all cursor-pointer"
                            >
                                <option value="anthropic">Anthropic</option>
                                <option value="openai">OpenAI</option>
                                <option value="gemini">Gemini</option>
                            </select>
                            <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-zinc-500">
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
                                >
                                    <path d="m6 9 6 6 6-6" />
                                </svg>
                            </div>
                        </div>
                    </div>

                    {/* Model Name Input */}
                    <div className="space-y-2">
                        <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                            Model Name
                        </label>
                        <input
                            type="text"
                            value={llm_info.model}
                            onChange={(e) => handleChange("model", e.target.value)}
                            placeholder="e.g. claude-3-opus-20240229"
                            className="w-full h-12 bg-zinc-900 border border-zinc-800 hover:border-zinc-700 rounded-lg px-4 text-zinc-200 placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all"
                        />
                    </div>

                    {/* API Key Input */}
                    <div className="space-y-2">
                        <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                            API Key
                        </label>
                        <input
                            type="password"
                            value={llm_info.apikey}
                            onChange={(e) => handleChange("apikey", e.target.value)}
                            placeholder="sk-..."
                            className="w-full h-12 bg-zinc-900 border border-zinc-800 hover:border-zinc-700 rounded-lg px-4 text-zinc-200 placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all font-mono text-sm"
                        />
                    </div>
                </div>

                {/* Actions */}
                <div className="pt-4 flex gap-4">
                    <button
                        onClick={handleReturn}
                        className="flex-1 h-12 bg-transparent border border-zinc-700 hover:border-zinc-500 hover:bg-zinc-900 text-zinc-300 font-bold rounded-lg transition-all"
                    >
                        Return
                    </button>
                    <button
                        onClick={handleSave}
                        disabled={saving}
                        className="flex-1 h-12 bg-zinc-100 hover:bg-white text-zinc-950 font-bold rounded-lg transition-all shadow-lg hover:shadow-zinc-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {saving ? "Saving..." : "Save Changes"}
                    </button>

                </div>
            </div>
        </main>
    );
}
