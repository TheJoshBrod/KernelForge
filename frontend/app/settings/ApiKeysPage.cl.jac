"""API Keys Page"""

import from "react-router-dom" { useNavigate }
import from react { useState }
import from ...components.backdrops.EmberBackground { EmberBackground }
import from ...components.settings.ApiKeysSettings { ApiKeysSettings }
import from ...components.settings.ApiKeysTestResults { ApiKeysTestResults }
import from ...components.ui.LoadingScreen { LoadingScreen }

import from ...context.ConfigContext { useConfig }

def:pub ApiKeysPage(props: any) -> Any {
    navigate = useNavigate();
    config_ctx = useConfig();
    config = config_ctx.config;
    setConfig = config_ctx.setConfig;
    saveConfig = config_ctx.saveConfig;
    loading = config_ctx.loading;

    saving_state = useState(False);
    saving = saving_state[0];
    setSaving = saving_state[1];

    testing_state = useState(False);
    testing = testing_state[0];
    setTesting = testing_state[1];

    test_results_state = useState(None);
    testResults = test_results_state[0];
    setTestResults = test_results_state[1];

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
        setSaving(True);
        await saveConfig(config);
        setSaving(False);
    }

    async def handleTestAll -> None {
        setTesting(True);
        setTestResults(None);
        try {
            resp = await __jacSpawn("TestAllApiKeys", "", {"cfg": config});
            data = resp.reports[0] if resp and resp.reports else {};
            setTestResults(data);
        } except Exception as e {
            console.error("Test failed:", e);
            setTestResults({"success": False, "error": str(e)});
        } finally {
            setTesting(False);
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
                        API Keys
                    </h1>
                    <p className="text-xs text-[var(--accent)] font-mono uppercase tracking-widest">
                        Store provider API keys globally. Model selection is per project.
                    </p>
                </div>
                
                <div className="flex-1 overflow-y-auto p-4 space-y-2 min-h-0 charcoal-scrollbar bg-[var(--bg-darker)]/30">
                    <ApiKeysSettings
                        config={config}
                        handleApiKeyChange={handleApiKeyChange}
                        handleSave={handleSave}
                        saving={saving}
                    />
                </div>
                
                <div className="p-6 border-t border-[var(--border)] bg-[var(--bg-panel)] shrink-0 flex justify-between items-center gap-4">
                    <button
                        onClick={handleTestAll}
                        disabled={testing or saving}
                        className={"px-6 h-10 w-full sm:w-auto flex justify-center items-center bg-zinc-800 hover:bg-zinc-700 text-zinc-200 font-bold rounded-lg transition-all border border-zinc-700 " + (
                            "opacity-50 cursor-not-allowed" if testing else ""
                        )}
                    >
                        {"Testing..." if testing else "Test All Keys"}
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
            <ApiKeysTestResults
                testResults={testResults}
                onClose={lambda -> None { setTestResults(None); }}
            />
        </main>;
}
