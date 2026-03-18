"""GPU Settings Page"""

import from react { useState, useEffect }
import from "react-router-dom" { useNavigate, useLocation }
import from ...components.backdrops.EmberBackground { EmberBackground }
import from ...components.ui.LoadingScreen { LoadingScreen }
import from ...context.ConfigContext { useConfig }

def:pub GpuPage -> Any {
    navigate = useNavigate();
    location = useLocation();
    config_ctx = useConfig();
    config = config_ctx.config;
    setConfig = config_ctx.setConfig;
    saveConfig = config_ctx.saveConfig;
    loading = config_ctx.loading;
    hwInfo = config_ctx.hwInfo;

    page_config_state = useState(config if config else {});
    pageConfig = page_config_state[0];
    setPageConfig = page_config_state[1];

    saving_state = useState(False);
    saving = saving_state[0];
    setSaving = saving_state[1];

    system_info_state = useState(hwInfo if hwInfo else {});  # Contains 'gpus' list
    systemInfo = system_info_state[0];
    setSystemInfo = system_info_state[1];

    config_selected_gpu = pageConfig["selected_gpu_info"]
    if "selected_gpu_info" in pageConfig
    else None;
    detected_gpus = systemInfo["gpus"]
    if systemInfo and "gpus" in systemInfo and systemInfo["gpus"]
    else (
        hwInfo["gpus"]
        if hwInfo and "gpus" in hwInfo and hwInfo["gpus"]
        else []
    );
    gpus = detected_gpus if detected_gpus and detected_gpus.length > 0 else (
        [config_selected_gpu] if config_selected_gpu else []
    );
    selected_idx = pageConfig["selected_gpu_index"]
    if "selected_gpu_index" in pageConfig
    else -1;
    has_explicit_selection = selected_idx != -1 and selected_idx < gpus.length;
    effective_idx = selected_idx if has_explicit_selection else (
        0 if gpus.length > 0 else -1
    );

    async def refreshGpuState -> None {
        try {
            active_idx = pageConfig["active_ssh_index"]
            if "active_ssh_index" in pageConfig
            else -1;
            ssh_connections = pageConfig["ssh_connections"]
            if "ssh_connections" in pageConfig
            else [];
            ssh_conn = None;
            if active_idx != -1 and active_idx < ssh_connections.length {
                ssh_conn = ssh_connections[active_idx];
            }

            payload = {};
            if ssh_conn {
                payload = {
                    "host": ssh_conn["host"] if "host" in ssh_conn else "",
                    "user": ssh_conn["user"] if "user" in ssh_conn else "",
                    "password": "",
                    "key_path": ssh_conn["key_path"] if "key_path" in ssh_conn else ""
                };
            }

            config_http = await fetch("/walker/GetConfig", {
                "method": "POST",
                "headers": {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                "body": JSON.stringify({})
            });
            config_payload = JSON.parse(await config_http.text());
            loaded_config = config_payload["data"]["reports"][0]
            if config_payload and "data" in config_payload and config_payload["data"]
            and "reports" in config_payload["data"] and config_payload["data"]["reports"]
            else {};

            info_http = await fetch("/walker/GetSystemInfo", {
                "method": "POST",
                "headers": {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                "body": JSON.stringify(payload)
            });
            info_payload = JSON.parse(await info_http.text());
            data = info_payload["data"]["reports"][0]
            if info_payload and "data" in info_payload and info_payload["data"]
            and "reports" in info_payload["data"] and info_payload["data"]["reports"]
            else None;

            if loaded_config {
                if (
                    ("selected_gpu_index" not in loaded_config or loaded_config["selected_gpu_index"] == -1)
                    and data and "gpus" in data and data["gpus"]
                ) {
                    loaded_config["selected_gpu_index"] = 0;
                    loaded_config["selected_gpu_info"] = data["gpus"][0];
                }
                setPageConfig(loaded_config);
                setConfig(loaded_config);
            }
            if data {
                setSystemInfo(data);
            }
        } except Exception as e {
            console.error("Failed to refresh GPU page state:", e);
            console.error("Error stack:", e.stack);
        }
    }

    useEffect(lambda -> None {
        if hwInfo and "gpus" in hwInfo and hwInfo["gpus"] and (
            not systemInfo
            or not ("gpus" in systemInfo and systemInfo["gpus"] and systemInfo["gpus"].length > 0)
        ) {
            setSystemInfo(hwInfo);
        }
    }, [hwInfo, systemInfo]);

    useEffect(lambda -> None { refreshGpuState();}, []);

    gpu_options = [];
    idx = 0;
    for g in gpus {
        base_name = g["name"] if "name" in g else f"GPU {idx}";
        conn = g["connection"] if "connection" in g else "local";
        gpu_name = f"{base_name} ({conn})"
        if conn != "local"
        else f"{base_name} (Local)";
        gpu_options.push(
            <option value={f"{idx}"} key={idx}>
                {gpu_name}
            </option>
        );
        idx += 1;
    }

    selected_gpu = gpus[effective_idx]
    if effective_idx != -1 and effective_idx < gpus.length
    else None;

    def handleGpuSelect(index: str) -> None {
        new_config = {** pageConfig};
        idx_int = parseInt(index);
        new_config["selected_gpu_index"] = idx_int;
        if idx_int != -1 and idx_int < gpus.length {
            new_config["selected_gpu_info"] = gpus[idx_int];
        } else {
            new_config["selected_gpu_info"] = None;
        }
        setPageConfig(new_config);
        setConfig(new_config);
    }

    async def handleSave -> None {
        setSaving(True);
        await saveConfig(pageConfig);
        setSaving(False);
    }

    def handleReturn -> None {
        return_to = location.state["return_to"] if location.state and "return_to" in location.state else None;
        if return_to and str(return_to).startsWith("/") {
            navigate(str(return_to));
        } else {
            navigate("/settings");
        }
    }

    if loading {
        return <LoadingScreen/>;
    }

    return
        <main className="relative h-screen w-screen overflow-hidden bg-[var(--bg-charcoal)] flex flex-col items-center justify-center font-sans text-white">
            <div className="absolute inset-0 z-0 industrial-grid-bg opacity-30"></div>
            <div className="noise-overlay opacity-30"></div>
            <EmberBackground className="absolute inset-0 z-0"/>
            <div className="absolute inset-0 bg-gradient-to-t from-black/50 via-transparent to-black/50 z-10 pointer-events-none"></div>

            <div className="relative z-20 flex flex-col items-center justify-center mb-4 animation-fade-in-up">
                <div className="flex items-center gap-4 mb-2">
                    <img src="/KernelForge.svg" alt="Kernel Forge Logo" className="w-12 h-12" />
                    <h1 className="text-3xl font-bold text-white tracking-wider uppercase industrial-font-display" style={{"textShadow": "0 0 20px rgba(225, 112, 85, 0.4)"}}>
                        Kernel Forge
                    </h1>
                </div>
            </div>

            <div className="relative z-10 w-full max-w-4xl h-[70vh] bg-[var(--bg-panel)]/90 backdrop-blur-md border border-[var(--border)] rounded-sm shadow-2xl flex flex-col animation-fade-in-up delay-1 clip-hex">
                <div className="p-6 border-b border-[var(--border)] space-y-4 shrink-0 relative overflow-hidden">
                    <div className="absolute top-0 left-0 right-0 heat-divider"></div>
                    <h1 className="text-2xl font-bold text-white tracking-wide uppercase industrial-font-display">
                        GPU Settings
                    </h1>
                    <p className="text-xs text-[var(--accent)] font-mono uppercase tracking-widest">
                        Select your active GPU
                    </p>
                </div>
                
                <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0 charcoal-scrollbar bg-[var(--bg-darker)]/30">
                    <div className="space-y-2">
                        <select
                            className="w-full h-10 bg-zinc-900 border border-zinc-800 rounded px-3 text-zinc-200"
                            value={f"{selected_idx}"}
                            onChange={lambda e: any  -> None { handleGpuSelect(
                                e.target.value
                            );}}
                        >
                            <option value="-1">
                                None
                            </option>
                            {gpu_options}
                        </select>
                    </div>
                    {(
                        <div
                            className="p-4 rounded-lg border border-zinc-800 bg-zinc-900/50 space-y-2"
                        >
                            <div className="text-sm font-bold text-zinc-100">
                                {"Selected GPU" if has_explicit_selection else "Detected GPU"}
                            </div>
                            {(
                                <div className="text-xs text-[var(--accent)] font-mono uppercase tracking-widest">
                                    No GPU is saved in config yet. Showing the first detected accelerator.
                                </div>
                            ) if selected_gpu and not has_explicit_selection else None}
                            <div className="text-sm text-zinc-300">
                                {selected_gpu["name"]
                                if selected_gpu
                                and "name" in selected_gpu
                                else "N/A"}
                            </div>
                            <div className="text-xs text-zinc-400 space-y-1">
                                <div>
                                    Memory:{selected_gpu["memory"]
                                    if selected_gpu
                                    and "memory" in selected_gpu
                                    else "N/A"}
                                </div>
                                <div>
                                    Compute:{selected_gpu["compute_capability"]
                                    if selected_gpu
                                    and "compute_capability" in selected_gpu
                                    else "N/A"}
                                </div>
                                <div>
                                    SMs:{selected_gpu["sm_count"]
                                    if selected_gpu
                                    and "sm_count" in selected_gpu
                                    else "N/A"}
                                </div>
                                <div>
                                    Runtime:{(
                                        "CUDA ready"
                                        if "runtime_usable" in selected_gpu and selected_gpu["runtime_usable"]
                                        else "Detected only"
                                    ) if selected_gpu
                                    else "N/A"}
                                </div>
                            </div>
                        </div>
                    )
                    if selected_gpu
                    else (
                        <div className="text-sm text-zinc-500 italic">
                            No accelerators detected.
                        </div>
                    )}
                </div>
                
                <div className="p-6 border-t border-[var(--border)] bg-[var(--bg-panel)] shrink-0 flex items-center justify-between gap-4">
                    <button
                        onClick={refreshGpuState}
                        className="px-6 h-10 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 font-bold rounded-lg transition-all border border-zinc-700"
                    >
                        Refresh
                    </button>
                    <div className="flex gap-4">
                        <button
                            onClick={handleReturn}
                            className="app-button"
                        >
                            Back
                        </button>
                        <button
                            onClick={handleSave}
                            disabled={saving}
                            className="app-button"
                        >
                            {("Saving..." if saving else "Save")}
                        </button>
                    </div>
                </div>
            </div>
        </main>;
}
