"""API Keys Page"""

import from react { useState }
import from "react-router-dom" { useNavigate }
import from "../../components/backdrops/GameOfLifeBackground.cl.jac" {
    GameOfLifeBackground
}
import from "../../components/settings/ApiKeysSettings.cl.jac" { ApiKeysSettings }
import from "../../context/ConfigContext.cl.jac" { useConfig }

def:pub ApiKeysPage {
    navigate = useNavigate();
    config_ctx = useConfig();
    config = config_ctx.config;
    setConfig = config_ctx.setConfig;
    saveConfig = config_ctx.saveConfig;
    loading = config_ctx.loading;

    has saving: bool = False;

    def handleReturn -> None {
        navigate("/settings");
    }

    def handleApiKeyChange(provider: str, value: str) -> None {
        new_api_keys = config["api_keys"] if "api_keys" in config else {};
        new_api_keys[provider] = value;

        # We need to update the entire config object
        new_config = {** config};
        new_config["api_keys"] = new_api_keys;

        setConfig(new_config);
    }

    async def handleSave -> None {
        saving = True;
        await saveConfig(config);
        saving = False;
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
                        API Keys
                    </h1>
                    <p className="settings-subtitle">
                        Manage your LLM provider keys
                    </p>
                </div>
                <ApiKeysSettings
                    config={config}
                    handleApiKeyChange={handleApiKeyChange}
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
