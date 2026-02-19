"""API Keys Page"""

import from "react-router-dom" { useNavigate }
import from ...components.backdrops.GameOfLifeBackground { GameOfLifeBackground }
import from ...components.settings.ApiKeysSettings { ApiKeysSettings }

import from ...context.ConfigContext { useConfig }

def:pub ApiKeysPage -> Any {
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
        api_keys = config["api_keys"] if config and "api_keys" in config else {};
        new_keys = {** api_keys};
        new_keys[provider] = value;
        new_config = {** config};
        new_config["api_keys"] = new_keys;
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
                        Store provider API keys globally. Model selection is per project.
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
