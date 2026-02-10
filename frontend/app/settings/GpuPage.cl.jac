"""GPU Settings Page"""

import from react { useState, useEffect }
import from "react-router-dom" { useNavigate }
import from "@jac/runtime" { jacSpawn }
import from "../../backdrops/GameOfLifeBackground.cl.jac" { GameOfLifeBackground }
import from "../../components/settings/GpuSettings.cl.jac" { GpuSettings }
import from "../../context/ConfigContext.cl.jac" { useConfig }

def:pub GpuPage {
    navigate = useNavigate();
    config_ctx = useConfig();
    config = config_ctx.config;
    setConfig = config_ctx.setConfig;
    saveConfig = config_ctx.saveConfig;
    loading = config_ctx.loading;

    has saving: bool = False,
        systemInfo: dict = {};  # Contains 'gpus' list

    async def fetchSystemInfo -> None {
        try {
            # Placeholder fetch - replace with actual walker if available
            resp = await jacSpawn("GetSystemInfo", "", {});
            data = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else None;
            if data {
                systemInfo = data;
            }
        } except (error) {
            console.error("Failed to fetch system info:", error);
        }
    }

    useEffect(lambda -> None { fetchSystemInfo();}, []);

    def handleGpuSelect(index: str) -> None {
        new_config = {** config};
        new_config["selected_gpu_index"] = int(index);
        setConfig(new_config);
    }

    async def handleSave -> None {
        saving = True;
        await saveConfig(config);
        saving = False;
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
                <GpuSettings
                    config={config}
                    systemInfo={systemInfo}
                    handleGpuSelect={handleGpuSelect}
                    handleSave={handleSave}
                    saving={saving}
                />
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
