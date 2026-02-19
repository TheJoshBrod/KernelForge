"""API Keys Page"""

import from "react-router-dom" { useNavigate }
import from react { useState }
import from ...components.backdrops.GameOfLifeBackground { GameOfLifeBackground }
import from ...components.settings.ApiKeysSettings { ApiKeysSettings }

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
                <div className="settings-header flex justify-between items-start">
                    <div>
                        <h1 className="settings-title">
                            API Keys
                        </h1>
                        <p className="settings-subtitle">
                            Store provider API keys globally. Model selection is per project.
                        </p>
                    </div>

                </div>

                <ApiKeysSettings
                    config={config}
                    handleApiKeyChange={handleApiKeyChange}
                    handleSave={handleSave}
                    saving={saving}
                />

                <div className="p-4 border-t border-zinc-800">
                    <div className="flex justify-between items-center mb-4">
                        <h3 className="text-sm font-medium text-zinc-300">Connection Test</h3>
                        <button
                            onClick={handleTestAll}
                            disabled={testing or saving}
                            className={"text-xs px-3 py-1.5 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-300 border border-zinc-700 transition-colors " + ("opacity-50 cursor-not-allowed" if testing else "")}
                        >
                            {"Testing..." if testing else "Test All Keys"}
                        </button>
                    </div>

                    {
                        <div className={"mt-4 p-3 rounded text-sm " + ("bg-green-900/20 border border-green-800 text-green-300" if testResults["success"] else "bg-red-900/20 border border-red-800 text-red-300")}>
                            <div className="flex items-center gap-2 mb-2">
                                <span className={("text-green-500" if testResults["success"] else "text-red-500")}>
                                    {"●" if testResults["success"] else "✖"}
                                </span>
                                <span className="font-medium">
                                    {"All Tests Passed" if testResults["success"] else "Tests Failed"}
                                </span>
                            </div>
                            {
                                <div className="space-y-1 pl-5">
                                    {Object.entries(testResults["results"]).map(lambda entry: any -> any {
                                        key = entry[0];
                                        val = entry[1];
                                        return (
                                            <div key={key} className="flex justify-between text-xs">
                                                <span className="capitalize text-zinc-400">{key}:</span>
                                                <span className={("text-green-400" if val["success"] else "text-red-400")}>
                                                    {val["latency_ms"] + "ms" if val["success"] else val["error"]}
                                                </span>
                                            </div>
                                        );
                                    })}
                                </div>
                                if "results" in testResults and testResults["results"]
                                else
                                <div className="pl-5 text-xs text-zinc-400">
                                    {testResults["error"] if "error" in testResults else ""}
                                </div>
                            }
                        </div>
                        if testResults
                        else null
                    }
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
