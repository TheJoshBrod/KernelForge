"""SSH Settings Page"""

import from react { useState }
import from "react-router-dom" { useNavigate }
import from ...components.backdrops.EmberBackground { EmberBackground }
import from ...components.ui.DesktopRuntime { isDesktopBuild }
import from ...components.ui.LoadingScreen { LoadingScreen }

import from .SSHSettings { SSHSettings }
import from ...context.ConfigContext { useConfig }

def:pub SshPage -> Any {
    navigate = useNavigate();
    config_ctx = useConfig();
    config = config_ctx.config;
    setConfig = config_ctx.setConfig;
    saveConfig = config_ctx.saveConfig;
    loading = config_ctx.loading;
    desktopBuild = isDesktopBuild();
    headingClass = "relative z-20 flex flex-col items-center justify-center mb-4";
    if not desktopBuild {
        headingClass += " animation-fade-in-up";
    }
    panelClass = "relative z-10 w-full max-w-4xl h-[70vh] border border-[var(--border)] rounded-sm shadow-2xl flex flex-col bg-[var(--bg-panel)]";
    if not desktopBuild {
        panelClass += " backdrop-blur-md animation-fade-in-up delay-1 clip-hex";
    }

    has saving: bool = False;

    view_state = useState("list");
    view = view_state[0];
    setView = view_state[1];

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
        return <LoadingScreen/>;
    }

    return
        <main className="relative h-screen w-screen overflow-hidden bg-[var(--bg-charcoal)] flex flex-col items-center justify-center font-sans text-white">
            <div className="absolute inset-0 z-0 industrial-grid-bg opacity-30"></div>
            {<div className="noise-overlay opacity-30"></div> if not desktopBuild else None}
            <EmberBackground className="absolute inset-0 z-0"/>
            <div className="absolute inset-0 bg-gradient-to-t from-black/50 via-transparent to-black/50 z-10 pointer-events-none"></div>

            <div className={headingClass}>
                <div className="flex items-center gap-4 mb-2">
                    <img src="/KernelForge.svg" alt="Kernel Forge Logo" className="w-12 h-12" />
                    <h1 className="text-3xl font-bold text-white tracking-wider uppercase industrial-font-display" style={{"textShadow": "0 0 20px rgba(225, 112, 85, 0.4)"}}>
                        Kernel Forge
                    </h1>
                </div>
            </div>

            <div className={panelClass}>
                <div className="p-6 border-b border-[var(--border)] space-y-4 shrink-0 relative overflow-hidden">
                    <div className="absolute top-0 left-0 right-0 heat-divider"></div>
                    <h1 className="text-2xl font-bold text-white tracking-wide uppercase industrial-font-display">
                        SSH Settings
                    </h1>
                    <p className="text-xs text-[var(--accent)] font-mono uppercase tracking-widest">
                        Manage remote connections
                    </p>
                </div>
                
                <div className="flex-1 overflow-hidden min-h-0 bg-[var(--bg-darker)]/30 flex flex-col">
                <SSHSettings
                    config={config}
                    handleSaveSshConnection={handleSaveSshConnection}
                    handleRemoveSshConnection={handleRemoveSshConnection}
                    handleSetActiveSshConnection={handleSetActiveSshConnection}
                    handleUpdateSshConnection={handleUpdateSshConnection}
                    saving={saving}
                    checkSshConnectionWalker={checkSshConnectionWalker}
                    view={view}
                    setView={setView}
                />
                </div>
                
                <div className={"p-6 border-t border-[var(--border)] bg-[var(--bg-panel)] shrink-0 flex items-center justify-between gap-4" + (" hidden" if view != "list" else "")}>
                    <button
                        onClick={lambda -> None { setView("add"); }}
                        className="px-6 h-10 w-full sm:w-auto flex justify-center items-center bg-zinc-800 hover:bg-zinc-700 text-zinc-200 font-bold rounded-lg transition-all border border-zinc-700"
                    >
                        + NEW SSH LINK
                    </button>
                    <div className="flex gap-4">
                        <button
                            onClick={handleReturn}
                            className="app-button"
                        >
                            Back
                        </button>
                    </div>
                </div>
            </div>
        </main>;
}
