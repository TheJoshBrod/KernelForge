"""Settings Page Component"""

import from react { useState, useEffect }
import from ..context.ConfigContext { useConfig }
import from ..assets.ChevronDownIcon { ChevronDownIcon }

def:pub Settings(onNavigate: any, onReturn: any = None) -> any {
    config_ctx = useConfig();
    config = config_ctx.config;
    setConfig = config_ctx.setConfig;
    saveConfig = config_ctx.saveConfig;
    loading = config_ctx.loading;

    saving_state = useState(False);
    saving = saving_state[0];
    setSaving = saving_state[1];

    useEffect(
        lambda -> None { def handleKeyDown(event: any) -> None {
            if event.key == "Escape" {
                if onReturn {
                    onReturn();
                } else {
                    onNavigate("/");
                }
            }
        } window.addEventListener("keydown", handleKeyDown);return lambda -> None { window.removeEventListener(
            "keydown", handleKeyDown
        );}; },
        [onNavigate, onReturn]
    );

    def handleChange(field: str, value: str) -> None {
        setConfig(
            lambda prev: any  -> any { new_llm_info = {** prev["llm_info"]};new_llm_info[
                field
            ] = value;return {** prev, "llm_info": new_llm_info}; }
        );
    }
    async def handleSave() -> None {
        setSaving(True);
        await saveConfig(config);
        setSaving(False);
        if onReturn {
            onReturn();
        } else {
            onNavigate("/");
        }
    }

    def handleReturn() -> None {
        if onReturn {
            onReturn();
        } else {
            onNavigate("/");
        }
    }

    if loading {
        return
            <div
                className="flex min-h-screen items-center justify-center bg-zinc-950 text-zinc-100"
            >
                <p>
                    Loading settings...
                </p>
            </div>;
    }

    llm_info = config["llm_info"];

    return
        <main
            className="flex min-h-screen flex-col items-center justify-center p-6 bg-zinc-900 text-zinc-100 selection:bg-zinc-700"
        >
            <div
                className="w-full max-w-md space-y-8 bg-zinc-950 p-8 rounded-xl border border-zinc-800 shadow-2xl"
            >
                <div className="text-center space-y-2">
                    <h1 className="text-3xl font-bold tracking-tight text-white">
                        Settings
                    </h1>
                    <p className="text-zinc-500 text-sm">
                        Configure your Model Provider (all information stored locally on your device)
                    </p>
                </div>
                <div className="space-y-6">
                    <div className="space-y-2">
                        <label
                            className="text-xs font-semibold text-zinc-400 uppercase tracking-wider"
                        >
                            Model Provider
                        </label>
                        <div className="relative">
                            <select
                                value={llm_info["provider"]}
                                onChange={lambda e: any  -> None { handleChange(
                                    "provider", e.target.value
                                );}}
                                className="w-full h-12 appearance-none bg-zinc-900 border border-zinc-800 hover:border-zinc-700 rounded-lg px-4 text-zinc-200 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all cursor-pointer"
                            >
                                <option value="anthropic">
                                    Anthropic
                                </option>
                                <option value="openai">
                                    OpenAI
                                </option>
                                <option value="gemini">
                                    Gemini
                                </option>
                            </select>
                            <div
                                className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-zinc-500"
                            >
                                <ChevronDownIcon />
                            </div>
                        </div>
                    </div>
                    <div className="space-y-2">
                        <label
                            className="text-xs font-semibold text-zinc-400 uppercase tracking-wider"
                        >
                            Model Name
                        </label>
                        <input
                            type="text"
                            value={llm_info["model"]}
                            onChange={lambda e: any  -> None { handleChange(
                                "model", e.target.value
                            );}}
                            placeholder="e.g. claude-3-opus-20240229"
                            className="w-full h-12 bg-zinc-900 border border-zinc-800 hover:border-zinc-700 rounded-lg px-4 text-zinc-200 placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all"
                        />
                    </div>
                    <div className="space-y-2">
                        <label
                            className="text-xs font-semibold text-zinc-400 uppercase tracking-wider"
                        >
                            API Key
                        </label>
                        <input
                            type="password"
                            value={llm_info["apikey"]}
                            onChange={lambda e: any  -> None { handleChange(
                                "apikey", e.target.value
                            );}}
                            placeholder="sk-..."
                            className="w-full h-12 bg-zinc-900 border border-zinc-800 hover:border-zinc-700 rounded-lg px-4 text-zinc-200 placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all font-mono text-sm"
                        />
                    </div>
                </div>
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
                        {("Saving..." if saving else "Save Changes")}
                    </button>
                </div>
            </div>
        </main>;
}
