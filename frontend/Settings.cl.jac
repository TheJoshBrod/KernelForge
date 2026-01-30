"""Settings Page Component"""

import from react { useEffect }
import from "react-router-dom" { useNavigate }
import from .ConfigContext { useConfig }

def:pub Settings -> any {
    navigate = useNavigate();
    config_ctx = useConfig();
    config = config_ctx.config;
    setConfig = config_ctx.setConfig;
    saveConfig = config_ctx.saveConfig;
    loading = config_ctx.loading;

    has saving: bool = False;

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

    def handleChange(field: str, value: str) -> None {
        new_llm_info = {
            "model": config["llm_info"]["model"],
            "apikey": config["llm_info"]["apikey"],
            "provider": config["llm_info"]["provider"]
        };
        new_llm_info[field] = value;
        setConfig({"llm_info": new_llm_info});
    }
    async def handleSave -> None {
        saving = True;
        await saveConfig(config);
        saving = False;
        navigate("/");
    }

    def handleReturn -> None {
        navigate("/");
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
    providers = config["providers"] if "providers" in config else [];
    providerOptions = [];
    for p in providers {
        providerOptions = providerOptions.concat([<option key={p} value={p}></option>]);
    }

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
                        Configure your Model Provider
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
                            <input
                                type="text"
                                list="provider-list"
                                value={llm_info["provider"]}
                                onChange={lambda e: any  -> None { handleChange(
                                    "provider", e.target.value
                                );}}
                                placeholder="openai, anthropic, gemini, ..."
                                className="w-full h-12 appearance-none bg-zinc-900 border border-zinc-800 hover:border-zinc-700 rounded-lg px-4 text-zinc-200 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all"
                            />
                            <datalist id="provider-list">
                                {providerOptions}
                            </datalist>
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
