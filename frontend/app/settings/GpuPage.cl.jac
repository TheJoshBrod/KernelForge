"""GPU Settings Page"""

import from react { useState, useEffect }
import from "react-router-dom" { useNavigate }
import from "@jac/runtime" { jacSpawn }
import from "../../components/backdrops/GameOfLifeBackground.cl.jac" {
    GameOfLifeBackground
}
import from "../../context/ConfigContext.cl.jac" { useConfig }

def:pub GpuPage {
    navigate = useNavigate();
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

            resp = await jacSpawn("GetSystemInfo", "", payload);

            data = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else None;

            if data {
                setSystemInfo(data);
            }
        } except (error) {
            console.error("Failed to fetch system info:", error);
            console.error("Error stack:", error.stack);
        }
    }

    useEffect(lambda -> None { fetchSystemInfo();}, [config]);

    gpu_options = [];
    idx = 0;
    for g in gpus {
        base_name = g["name"] if "name" in g else f"GPU {idx}";
        conn = g["connection"] if "connection" in g else "local";
        gpu_name = f"{base_name} ({conn})" if conn != "local" else f"{base_name} (Local)";
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
        navigate("/settings");
    }

    if loading {
        return
            <div className="app-shell app-shell--center settings-shell">
                <GameOfLifeBackground />
                <div className="app-backdrop"></div>
                <div className="app-content-layer">
                    <p>
                        Loading...
                    </p>
                </div>
            </div>;
    }

    return
        <main className="app-shell app-shell--center settings-shell">
            <GameOfLifeBackground />
            <div className="app-backdrop"></div>
            <div className="settings-card panel app-content-layer">
                <div className="settings-header">
                    <h1 className="settings-title">
                        GPU Settings
                    </h1>
                    <p className="settings-subtitle">
                        Select your active GPU
                    </p>
                </div>
                <div className="space-y-4">
                    <div className="space-y-2">
                        <div className="text-sm font-semibold text-zinc-300">
                            Available GPUs {(
                                "(local + connected SSH)" if gpus.length > 0 else ""
                            )}
                        </div>
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
                                    Memory: {gpus[selected_idx]["memory"]
                                    if selected_idx != -1
                                    and selected_idx < gpus.length
                                    and "memory" in gpus[selected_idx]
                                    else "N/A"}
                                </div>
                                <div>
                                    Compute: {gpus[selected_idx]["compute_capability"]
                                    if selected_idx != -1
                                    and selected_idx < gpus.length
                                    and "compute_capability" in gpus[selected_idx]
                                    else "N/A"}
                                </div>
                                <div>
                                    SMs: {gpus[selected_idx]["sm_count"]
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
                    <div className="flex gap-3">
                        <button
                            onClick={fetchSystemInfo}
                            className="flex-1 h-10 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 font-bold rounded-lg transition-all border border-zinc-700"
                        >
                            Refresh
                        </button>
                        <button
                            onClick={handleSave}
                            disabled={saving}
                            className="flex-1 h-10 bg-zinc-100 hover:bg-white text-zinc-950 font-bold rounded-lg transition-all shadow-lg hover:shadow-zinc-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {"Saving..." if saving else "Save"}
                        </button>
                    </div>
                </div>
                <div className="settings-actions-block">
                    <div className="settings-actions">
                        <button
                            onClick={handleReturn}
                            className="app-button app-button--ghost app-button--wide"
                        >
                            Back
                        </button>
                    </div>
                </div>
            </div>
        </main>;
}
