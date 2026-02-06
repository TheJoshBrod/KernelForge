"""Settings Page Component"""

import from react { useEffect }
import from "@jac/runtime" { jacSpawn }
import from "react-router-dom" { useNavigate }
import from "backdrops/GameOfLifeBackground" { GameOfLifeBackground }
import from .ConfigContext { useConfig }

def:pub Settings -> any {
    navigate = useNavigate();
    config_ctx = useConfig();
    config = config_ctx.config;
    setConfig = config_ctx.setConfig;
    saveConfig = config_ctx.saveConfig;
    loading = config_ctx.loading;

    has saving: bool = False,
        apiTesting: bool = False,
        apiTest: dict = {};

    def getReturnPath -> str {
        try {
            search = window.location.search
            if window and window.location and "search" in window.location
            else "";
            params = URLSearchParams(search);
            candidate = params.get("returnTo");
            if candidate
            and candidate.startswith("/")
            and not candidate.startswith("//")
            and candidate != "/settings" {
                return candidate;
            }
        } except Exception as e {
            return "/";
        }
        return "/";
    }

    def handleKeyDown(event: any) -> None {
        if event.key == "Escape" {
            navigate(getReturnPath());
        }
    }

    useEffect(
        lambda -> None { window.addEventListener("keydown", handleKeyDown);return lambda -> None { window.removeEventListener(
            "keydown", handleKeyDown
        );}; },
        []
    );

    def handleChange(field: str, value: str) -> None {
        new_llm_info = {
            "model": config["llm_info"]["model"],
            "apikey": config["llm_info"]["apikey"],
            "provider": config["llm_info"]["provider"]
        };
        new_llm_info[field] = value;
        setConfig({"llm_info": new_llm_info});
    }
    async def handleSave -> None {
        saving = True;
        await saveConfig(config);
        saving = False;
        navigate(getReturnPath());
    }

    async def handleTest -> None {
        apiTesting = True;
        apiTest = {};
        try {
            resp = await jacSpawn("TestLLMConnection", "", {"cfg": config});
            data = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else None;
            if data {
                apiTest = data;
            } else {
                apiTest = {"success": False, "error": "No response from server"};
            }
        } except (error) {
            console.error("API test failed:", error);
            apiTest = {"success": False, "error": "API test failed"};
        } finally {
            apiTesting = False;
        }
    }

    def handleReturn -> None {
        navigate(getReturnPath());
    }

    if loading {
        return
            <div className="app-shell app-shell--center settings-shell">
                <GameOfLifeBackground />
                <div className="app-backdrop"></div>
                <div className="app-content-layer">
                    <p>
                        Loading settings...
                    </p>
                </div>
            </div>;
    }

    llm_info = config["llm_info"];
    providers = config["providers"] if "providers" in config else [];
    providerOptions = [];
    for p in providers {
        providerOptions = providerOptions.concat([<option key={p} value={p}></option>]);
    }

    testMessage = "";
    testTone = "tone-muted";
    hasLatency = False;
    latencyValue = 0;
    hasResponse = False;
    responseValue = "";
    if apiTest and "success" in apiTest {
        if apiTest["success"] {
            testTone = "tone-success";
            testMessage = "Success";
            if "latency_ms" in apiTest {
                hasLatency = True;
                latencyValue = apiTest["latency_ms"];
            }
            if "response" in apiTest and apiTest["response"] {
                hasResponse = True;
                responseValue = apiTest["response"];
            }
        } else {
            testTone = "tone-error";
            testMessage = apiTest["error"] if "error" in apiTest else "API test failed";
        }
    }

    return
        <main className="app-shell app-shell--center settings-shell">
            <GameOfLifeBackground />
            <div className="app-backdrop"></div>
            <div className="settings-card panel app-content-layer">
                <div className="settings-header">
                    <h1 className="settings-title">
                        Settings
                    </h1>
                    <p className="settings-subtitle">
                        Configure your Model Provider
                    </p>
                </div>
                <div className="form-stack">
                    <div className="form-field">
                        <label className="input-label">
                            Model Provider
                        </label>
                        <div className="relative">
                            <input
                                type="text"
                                list="provider-list"
                                value={llm_info["provider"]}
                                onChange={lambda e: any  -> None { handleChange(
                                    "provider", e.target.value
                                );}}
                                placeholder="openai, anthropic, gemini, ..."
                                className="input"
                            />
                            <datalist id="provider-list">
                                {providerOptions}
                            </datalist>
                        </div>
                    </div>
                    <div className="form-field">
                        <label className="input-label">
                            Model Name
                        </label>
                        <input
                            type="text"
                            value={llm_info["model"]}
                            onChange={lambda e: any  -> None { handleChange(
                                "model", e.target.value
                            );}}
                            placeholder="e.g. claude-3-opus-20240229"
                            className="input"
                        />
                    </div>
                    <div className="form-field">
                        <label className="input-label">
                            API Key
                        </label>
                        <input
                            type="password"
                            value={llm_info["apikey"]}
                            onChange={lambda e: any  -> None { handleChange(
                                "apikey", e.target.value
                            );}}
                            placeholder="sk-..."
                            className="input input--mono"
                        />
                    </div>
                </div>
                <div className="settings-actions-block">
                    <button
                        onClick={handleTest}
                        disabled={apiTesting}
                        className="app-button app-button--secondary app-button--full"
                    >
                        {("Testing..." if apiTesting else "Test API Connection")}
                    </button>
                    {testMessage
                    and <div className={"settings-test " + testTone}>
                        <span>
                            {testMessage}
                        </span>
                        {hasLatency
                        and <span>
                            {" - "}{latencyValue}ms
                        </span>}
                        {hasResponse
                        and <span>
                            {" - "}{responseValue}
                        </span>}
                    </div>}
                    <div className="settings-actions">
                        <button
                            onClick={handleReturn}
                            className="app-button app-button--ghost app-button--wide"
                        >
                            Return
                        </button>
                        <button
                            onClick={handleSave}
                            disabled={saving}
                            className="app-button app-button--primary app-button--wide"
                        >
                            {("Saving..." if saving else "Save Changes")}
                        </button>
                    </div>
                </div>
            </div>
        </main>;
}
