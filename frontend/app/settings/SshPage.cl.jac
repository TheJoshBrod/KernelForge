"""SSH Settings Page"""

import from react { useState }
import from "react-router-dom" { useNavigate }
import from ...components.backdrops.GameOfLifeBackground { GameOfLifeBackground }
import from ...components.settings.SettingsMenu { SettingsMenu }
import from .SSHSettings { SSHSettings }
import from ...context.ConfigContext { useConfig }

def:pub SshPage {
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

    def handleSaveSshConnection(connection: dict) -> None {
        new_connections = config["ssh_connections"]
        if "ssh_connections" in config
        else [];

        new_connections.push(connection);

        new_config = {** config};
        new_config["ssh_connections"] = new_connections;
        setConfig(new_config);

        saveConfig(new_config);
    }

    def handleRemoveSshConnection(index: int) -> None {
        new_connections = config["ssh_connections"]
        if "ssh_connections" in config
        else [];
        new_connections.splice(index, 1);

        new_config = {** config};
        new_config["ssh_connections"] = new_connections;

        # Adjust active index if needed
        active_idx = config["active_ssh_index"] if "active_ssh_index" in config else -1;
        if active_idx == index {
            # Deleted the active connection -> clear active
            new_config["active_ssh_index"] = -1;
        } elif active_idx > index {
            # Shift active index down by one when removing earlier entry
            new_config["active_ssh_index"] = active_idx - 1;
        }

        setConfig(new_config);
        saveConfig(new_config);
    }

    def handleSetActiveSshConnection(index: int) -> None {
        new_config = {** config};
        new_config["active_ssh_index"] = index;
        setConfig(new_config);
        # Auto save selection
        saveConfig(new_config);
    }

    def handleUpdateSshConnection(index: int, updates: dict) -> None {
        new_connections = config["ssh_connections"]
        if "ssh_connections" in config
        else [];
        if index >= 0 and index < new_connections.length {
            # Update specific fields
            updated_conn = {** new_connections[index], ** updates};
            new_connections[index] = updated_conn;
            new_config = {** config};
            new_config["ssh_connections"] = new_connections;
            setConfig(new_config);
            saveConfig(new_config);
        }
    }

    async def checkSshConnectionWalker(
        host: str, user: str, password: str, key_path: str
    ) -> dict {
        try {
            resp = await __jacSpawn(
                "CheckSshConnection",
                "",
                {
                    "host": host,
                    "user": user,
                    "password": password,
                    "key_path": key_path
                }
            );
            data = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else None;

            if data {
                return data;
            }
            return {"success": False, "error": "No response"};
        } except Exception as e {
            console.error("SSH Check failed", e);
            return {"success": False, "error": str(e)};
        }
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
            <div className="settings-card panel app-content-layer !max-w-7xl w-full">
                <div className="settings-header">
                    <h1 className="settings-title">
                        SSH Settings
                    </h1>
                    <p className="settings-subtitle">
                        Manage remote connections
                    </p>
                </div>
                <SettingsMenu />
                <SSHSettings
                    config={config}
                    handleSaveSshConnection={handleSaveSshConnection}
                    handleRemoveSshConnection={handleRemoveSshConnection}
                    handleSetActiveSshConnection={handleSetActiveSshConnection}
                    handleUpdateSshConnection={handleUpdateSshConnection}
                    saving={saving}
                    checkSshConnectionWalker={checkSshConnectionWalker}
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
