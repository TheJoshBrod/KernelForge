"""Settings Page Component"""

import from react { useState, useEffect }
import from ..context.ConfigContext { useConfig }
import from ..assets.ChevronDownIcon { ChevronDownIcon }
import from ..assets.ArrowRightIcon { ArrowRightIcon }

def:pub Settings(onNavigate: any, onReturn: any = None, getSystemInfoWalker: any = None) -> any {
    config_ctx = useConfig();
    config = config_ctx.config;
    setConfig = config_ctx.setConfig;
    saveConfig = config_ctx.saveConfig;
    loading = config_ctx.loading;

    # View state: "main", "api_keys", "gpu"
    view_state = useState("main");
    view = view_state[0];
    setView = view_state[1];

    saving_state = useState(False);
    saving = saving_state[0];
    setSaving = saving_state[1];

    system_info_state = useState({});
    systemInfo = system_info_state[0];
    setSystemInfo = system_info_state[1];

    # Initialize api_keys if missing
    useEffect(
        lambda -> None {
            if config and not "api_keys" in config {
                 setConfig(lambda prev: any -> any { return {**prev, "api_keys": {"openai": "", "gemini": "", "anthropic": ""}}; });
            }
            if config and not "selected_gpu_index" in config {
                 setConfig(lambda prev: any -> any { return {**prev, "selected_gpu_index": 0}; });
            }
        }, [config]
    );

    # Fetch system info when entering GPU view
    useEffect(
        lambda -> None {
            async def fetchSystemInfo -> None {
                if view == "gpu" and getSystemInfoWalker {
                    info = await getSystemInfoWalker();
                    setSystemInfo(info);
                }
            }
            fetchSystemInfo();
        }, [view]
    );

    def handleApiKeyChange(provider: str, value: str) -> None {
        setConfig(
            lambda prev: any -> any {
                current_keys = prev["api_keys"] if "api_keys" in prev else {};
                new_keys = {**current_keys};
                new_keys[provider] = value;
                return {**prev, "api_keys": new_keys};
            }
        );
    }

    async def handleSave() -> None {
        setSaving(True);
        await saveConfig(config);
        setSaving(False);
        setView("main");
    }

    def handleGpuSelect(index: int) -> None {
        setConfig(lambda prev: any -> any { return {**prev, "selected_gpu_index": int(index)}; });
    }

    def handleReturn() -> None {
        if view == "main" {
            if onReturn { onReturn(); } else { onNavigate("/"); }
        } else {
            setView("main");
        }
    }

    if loading {
        return <div className="flex min-h-screen items-center justify-center bg-zinc-950 text-zinc-100"><p>Loading settings...</p></div>;
    }

    return
        <main className="flex min-h-screen flex-col items-center justify-center p-6 bg-zinc-900 text-zinc-100 selection:bg-zinc-700">
            <div className="w-full max-w-md space-y-8 bg-zinc-950 p-8 rounded-xl border border-zinc-800 shadow-2xl">
                <div className="text-center space-y-2">
                    <h1 className="text-3xl font-bold tracking-tight text-white">Settings</h1>
                    <p className="text-zinc-500 text-sm">
                        {(
                            "Configure your experience"
                            if view == "main"
                            else "Manage API Credentials"
                            if view == "api_keys"
                            else "Hardware Configuration"
                        )}
                    </p>
                </div>

                {
                    # MAIN VIEW
                    <div className="space-y-4">
                        <button
                            onClick={lambda -> None { setView("api_keys"); }}
                            className="w-full h-16 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 hover:border-zinc-600 rounded-lg px-6 flex items-center justify-between group transition-all"
                        >
                            <span className="font-semibold text-zinc-300 group-hover:text-white">API Keys</span>
                            <ArrowRightIcon className="text-zinc-500 group-hover:text-white transition-colors" />
                        </button>
                        <button
                            onClick={lambda -> None { setView("gpu"); }}
                            className="w-full h-16 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 hover:border-zinc-600 rounded-lg px-6 flex items-center justify-between group transition-all"
                        >
                            <span className="font-semibold text-zinc-300 group-hover:text-white">GPU Settings</span>
                            <ArrowRightIcon className="text-zinc-500 group-hover:text-white transition-colors" />
                        </button>
                    </div>
                    if view == "main"

                    # API KEYS VIEW
                    else <div className="space-y-6">
                        <div className="space-y-4">
                            {
                                [<div className="space-y-2" key={provider}>
                                    <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">{provider}</label>
                                    <input
                                        type="password"
                                        value={config["api_keys"][provider] if "api_keys" in config and provider in config["api_keys"] else ""}
                                        onChange={lambda e: any -> None { handleApiKeyChange(provider, e.target.value); }}
                                        placeholder={f"Enter {provider} API Key"}
                                        className="w-full h-12 bg-zinc-900 border border-zinc-800 hover:border-zinc-700 rounded-lg px-4 text-zinc-200 placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all font-mono text-sm"
                                    />
                                </div> for provider in ["openai", "gemini", "anthropic"]]
                            }
                        </div>
                        <button
                            onClick={handleSave}
                            disabled={saving}
                            className="w-full h-12 bg-zinc-100 hover:bg-white text-zinc-950 font-bold rounded-lg transition-all shadow-lg hover:shadow-zinc-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {("Saving..." if saving else "Save Keys")}
                        </button>
                    </div>
                    if view == "api_keys"

                    # GPU SETTINGS VIEW
                    else <div className="space-y-6">
                        <div className="space-y-2">
                             <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Select Active GPU</label>
                             <div className="relative">
                                <select
                                    value={config["selected_gpu_index"] if "selected_gpu_index" in config else 0}
                                    onChange={lambda e: any -> None { handleGpuSelect(parseInt(e.target.value)); }}
                                    className="w-full h-12 appearance-none bg-zinc-900 border border-zinc-800 hover:border-zinc-700 rounded-lg px-4 text-zinc-200 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all cursor-pointer"
                                    disabled={not systemInfo or not "gpus" in systemInfo}
                                >
                                    {(
                                        systemInfo["gpus"].map(lambda gpu: any, i: any -> any:
                                            <option value={i} key={gpu["name"]}>{f"{i}: {gpu['name']}"}</option>
                                        )
                                        if systemInfo and "gpus" in systemInfo
                                        else <option>Loading GPUs...</option>
                                    )}
                                </select>
                                <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-zinc-500">
                                    <ChevronDownIcon />
                                </div>
                             </div>
                        </div>

                        {(
                            <div className="p-4 bg-zinc-900/50 rounded-lg border border-zinc-800 space-y-2">
                                <h3 className="text-sm font-medium text-zinc-300">Selected GPU Stats</h3>
                                <div className="grid grid-cols-[auto_1fr] gap-4 text-sm">
                                    <div className="text-zinc-500">Name</div>
                                    <div className="text-zinc-200 text-right whitespace-nowrap overflow-hidden text-ellipsis pl-2" title={systemInfo["gpus"][config["selected_gpu_index"]]["name"]}>
                                        {systemInfo["gpus"][config["selected_gpu_index"]]["name"]}
                                    </div>
                                    <div className="text-zinc-500">Memory</div>
                                    <div className="text-zinc-200 text-right">{systemInfo["gpus"][config["selected_gpu_index"]]["memory"]}</div>
                                    <div className="text-zinc-500">Architecture Version</div>
                                    <div className="text-zinc-200 text-right">{systemInfo["gpus"][config["selected_gpu_index"]]["compute_capability"]}</div>
                                    <div className="text-zinc-500">SM Count</div>
                                    <div className="text-zinc-200 text-right">{systemInfo["gpus"][config["selected_gpu_index"]]["sm_count"]}</div>
                                </div>
                            </div>
                            if systemInfo and "gpus" in systemInfo and config["selected_gpu_index"] < systemInfo["gpus"].length
                            else null
                        )}
                        <button
                            onClick={handleSave}
                            disabled={saving}
                            className="w-full h-12 bg-zinc-100 hover:bg-white text-zinc-950 font-bold rounded-lg transition-all shadow-lg hover:shadow-zinc-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {("Saving..." if saving else "Save Selection")}
                        </button>
                    </div>
                    if view == "gpu"
                    else null
                }

                <div className="pt-4">
                    <button
                        onClick={handleReturn}
                        className="w-full h-12 bg-transparent border border-zinc-700 hover:border-zinc-500 hover:bg-zinc-900 text-zinc-300 font-bold rounded-lg transition-all"
                    >
                        {( "Back" if view != "main" else "Return to Dashboard" )}
                    </button>
                </div>
            </div>
        </main>;
}
