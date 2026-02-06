"""Settings Page Component"""

import from react { useState, useEffect }
import from ..context.ConfigContext { useConfig }
import from .settings.ApiKeysSettings { ApiKeysSettings }
import from .settings.GpuSettings { GpuSettings }
import from .settings.SSHSettings { SSHSettings }
import from .settings.SettingsMenu { SettingsMenu }

def:pub Settings(onNavigate: any, onReturn: any = None, getSystemInfoWalker: any = None, checkSshConnectionWalker: any = None) -> any {
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
            if config and not "ssh_connections" in config {
                 setConfig(lambda prev: any -> any { return {**prev, "ssh_connections": []}; });
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

    async def handleSaveSshConnection(connection: any) -> None {
        setSaving(True);
        try {
            current_conns = config["ssh_connections"] if "ssh_connections" in config else [];
            
            # Simple replace logic: Filter out matching host/user, then append new
            host = connection["host"] if "host" in connection else "";
            user = connection["user"] if "user" in connection else "";
            
            if host and user {
                new_conns = [c for c in current_conns if not (("host" in c and c["host"] == host) and ("user" in c and c["user"] == user))];
                new_conns.push(connection);
                
                new_config = {**config, "ssh_connections": new_conns};
                setConfig(new_config);
                await saveConfig(new_config);
            }
        } finally {
            setSaving(False);
        }
    }

    async def handleRemoveSshConnection(index: int) -> None {
        setSaving(True);
        try {
            current_conns = config["ssh_connections"] if "ssh_connections" in config else [];
            if index >= 0 and index < current_conns.length {
                new_conns = current_conns.filter(lambda c: any, i: int -> bool { return i != index; });
                
                new_config = {**config, "ssh_connections": new_conns};
                setConfig(new_config);
                await saveConfig(new_config);
            }
        } finally {
            setSaving(False);
        }
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
        <main className="flex min-h-screen flex-col items-center justify-center p-4 bg-zinc-900 text-zinc-100 selection:bg-zinc-700">
            <div className="w-full max-w-md space-y-4 bg-zinc-950 p-6 rounded-xl border border-zinc-800 shadow-2xl">
                <div className="text-center space-y-2">
                    <h1 className="text-3xl font-bold tracking-tight text-white">Settings</h1>
                    <p className="text-zinc-500 text-sm">
                        {(
                            "Configure your experience"
                            if view == "main"
                            else "Manage API Credentials"
                            if view == "api_keys"
                            else "Hardware Configuration"
                            if view == "gpu"
                            else "SSH Configuration"
                        )}
                    </p>
                </div>

                {
                    # MAIN VIEW
                    <SettingsMenu setView={setView} />
                    if view == "main"

                    # API KEYS VIEW
                    else <ApiKeysSettings
                        config={config}
                        handleApiKeyChange={handleApiKeyChange}
                        handleSave={handleSave}
                        saving={saving}
                    />
                    if view == "api_keys"

                    # GPU SETTINGS VIEW
                    else <GpuSettings
                        config={config}
                        systemInfo={systemInfo}
                        handleGpuSelect={handleGpuSelect}
                        handleSave={handleSave}
                        saving={saving}
                    />
                    if view == "gpu"

                    # SSH SETTINGS VIEW
                    else <SSHSettings
                        config={config}
                        handleSaveSshConnection={handleSaveSshConnection}
                        handleRemoveSshConnection={handleRemoveSshConnection}
                        saving={saving}
                        checkSshConnectionWalker={checkSshConnectionWalker}
                    />
                    if view == "ssh"
                    else null
                }

                <div className="pt-2">
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
