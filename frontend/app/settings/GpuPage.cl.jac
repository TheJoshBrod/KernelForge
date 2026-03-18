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

    saving_state = useState(False);
    saving = saving_state[0];
    setSaving = saving_state[1];

    system_info_state = useState({});  # Contains 'gpus' list
    systemInfo = system_info_state[0];
    setSystemInfo = system_info_state[1];

    gpus = systemInfo["gpus"] if systemInfo and "gpus" in systemInfo else [];
    selected_idx = config["selected_gpu_index"]
    if "selected_gpu_index" in config
    else -1;

    async def fetchSystemInfo -> None {
        try {
            active_idx = config["active_ssh_index"]
            if "active_ssh_index" in config
            else -1;
            ssh_connections = config["ssh_connections"]
            if "ssh_connections" in config
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

            resp = await __jacSpawn("GetSystemInfo", "", payload);

            data = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else None;

            if data {
                setSystemInfo(data);
            }
        } except Exception as e {
            console.error("Failed to fetch system info:", e);
            console.error("Error stack:", e.stack);
        }
    }

    useEffect(lambda -> None { fetchSystemInfo();}, [config]);

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

    selected_gpu = gpus[selected_idx]
    if selected_idx != -1 and selected_idx < gpus.length
    else None;

    def handleGpuSelect(index: str) -> None {
        new_config = {** config};
        idx_int = parseInt(index);
        new_config["selected_gpu_index"] = idx_int;
        if idx_int != -1 and idx_int < gpus.length {
            new_config["selected_gpu_info"] = gpus[idx_int];
        } else {
            new_config["selected_gpu_info"] = None;
        }
        setConfig(new_config);
    }

    async def handleSave -> None {
        setSaving(True);
        await saveConfig(config);
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
                                Selected GPU
                            </div>
                            <div className="text-sm text-zinc-300">
                                {gpus[selected_idx]["name"]
                                if selected_idx != -1
                                and selected_idx < gpus.length
                                and "name" in gpus[selected_idx]
                                else "N/A"}
                            </div>
                            <div className="text-xs text-zinc-400 space-y-1">
                                <div>
                                    Memory:{gpus[selected_idx]["memory"]
                                    if selected_idx != -1
                                    and selected_idx < gpus.length
                                    and "memory" in gpus[selected_idx]
                                    else "N/A"}
                                </div>
                                <div>
                                    Compute:{gpus[selected_idx]["compute_capability"]
                                    if selected_idx != -1
                                    and selected_idx < gpus.length
                                    and "compute_capability" in gpus[selected_idx]
                                    else "N/A"}
                                </div>
                                <div>
                                    SMs:{gpus[selected_idx]["sm_count"]
                                    if selected_idx != -1
                                    and selected_idx < gpus.length
                                    and "sm_count" in gpus[selected_idx]
                                    else "N/A"}
                                </div>
                            </div>
                        </div>
                    )
                    if selected_idx != -1 and selected_idx < gpus.length
                    else (
                        <div className="text-sm text-zinc-500 italic">
                            Select a GPU to see stats.
                        </div>
                    )}
                </div>
                
                <div className="p-6 border-t border-[var(--border)] bg-[var(--bg-panel)] shrink-0 flex items-center justify-between gap-4">
                    <button
                        onClick={fetchSystemInfo}
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
