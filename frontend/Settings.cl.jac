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
        authBusy: bool = False,
        apiKeyDraft: str = "",
        apiTest: dict = {},
        authStatus: dict = {},
        authMessage: str = "";

    def _copy_cfg(src: dict) -> dict {
        out = {};
        for key in src {
            out[key] = src[key];
        }
        return out;
    }

    def _auth_cfg(cfg: dict) -> dict {
        auth = cfg["auth"] if cfg and "auth" in cfg and cfg["auth"] else {};
        if not auth {
            auth = {
                "mode": "auto",
                "provider": "openai",
                "model": "",
                "api_key_saved": False,
                "account_session_detected": False,
                "last_checked_at": 0
            };
        }
        return auth;
    }

    async def refreshAuthStatus -> None {
        authBusy = True;
        try {
            resp = await jacSpawn("GetAuthStatus", "", {});
            data = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else {};
            authStatus = data if data else {};
            if data and "auth" in data {
                mode_effective = data["auth"]["mode_effective"]
                if "mode_effective" in data["auth"]
                else "unknown";
                reason = data["auth"]["reason"] if "reason" in data["auth"] else "";
                authMessage = ("Auth: " + mode_effective + (" (" + reason + ")" if reason else ""));
            } else {
                authMessage = "Auth status unavailable";
            }
        } except (error) {
            authStatus = {"success": False, "error": "Failed to detect auth status"};
            authMessage = "Auth status check failed";
        } finally {
            authBusy = False;
        }
    }

    def _return_cache_key() -> str {
        return "cgins_settings_return_to";
    }

    def _is_valid_return(candidate: str) -> bool {
        return candidate
        and candidate.startswith("/")
        and not candidate.startswith("//")
        and candidate != "/settings";
    }

    def getReturnPath -> str {
        try {
            search = window.location.search
            if window and window.location and "search" in window.location
            else "";
            params = URLSearchParams(search);
            candidate = params.get("returnTo");
            if _is_valid_return(candidate) {
                window.sessionStorage.setItem(_return_cache_key(), candidate);
                return candidate;
            }
            cached = window.sessionStorage.getItem(_return_cache_key());
            if _is_valid_return(cached) {
                return cached;
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

    useEffect(lambda -> None { refreshAuthStatus();}, []);

    def handleChange(field: str, value: str) -> None {
        new_llm_info = {
            "model": config["llm_info"]["model"],
            "apikey": "",
            "provider": config["llm_info"]["provider"]
        };
        new_llm_info[field] = value;

        new_auth = _auth_cfg(config);
        if field == "provider" or field == "model" {
            new_auth[field] = value;
        }

        next_cfg = _copy_cfg(config);
        next_cfg["llm_info"] = new_llm_info;
        next_cfg["auth"] = new_auth;
        setConfig(next_cfg);
    }

    def handleAuthModeChange(value: str) -> None {
        next_cfg = _copy_cfg(config);
        auth = _auth_cfg(config);
        auth["mode"] = value;
        next_cfg["auth"] = auth;
        setConfig(next_cfg);
    }

    async def handleSaveApiKey -> None {
        if not apiKeyDraft {
            authMessage = "Enter an API key first.";
            return;
        }
        authBusy = True;
        try {
            llm = config["llm_info"] if config and "llm_info" in config else {};
            provider = llm["provider"] if llm and "provider" in llm else "openai";
            model = llm["model"] if llm and "model" in llm and llm["model"] else "gpt-5.2";
            resp = await jacSpawn(
                "SaveAuthSecret",
                "",
                {"provider": provider, "model": model, "apiKey": apiKeyDraft}
            );
            data = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else {};
            if data and "success" in data and data["success"] {
                apiKeyDraft = "";
                authMessage = "API key saved to keychain.";
                await refreshAuthStatus();
            } else {
                authMessage = (
                    data["error"]
                    if data and "error" in data and data["error"]
                    else "Failed to save API key."
                );
            }
        } except (error) {
            console.error("Failed to save API key:", error);
            authMessage = "Failed to save API key.";
        } finally {
            authBusy = False;
        }
    }

    async def handleSave -> None {
        saving = True;
        next_cfg = _copy_cfg(config);
        llm = next_cfg["llm_info"] if next_cfg and "llm_info" in next_cfg else {};
        llm["apikey"] = "";
        next_cfg["llm_info"] = llm;
        await saveConfig(next_cfg);
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

    async def handleDetectAccountSession -> None {
        await refreshAuthStatus();
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
    auth = _auth_cfg(config);
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
                            Auth Mode
                        </label>
                        <select
                            value={auth["mode"]}
                            onChange={lambda e: any  -> None { handleAuthModeChange(e.target.value);}}
                            className="input"
                        >
                            <option value="auto">
                                auto
                            </option>
                            <option value="api_key">
                                api_key
                            </option>
                            <option value="account_session">
                                account_session
                            </option>
                        </select>
                    </div>
                    <div className="form-field">
                        <label className="input-label">
                            API Key
                        </label>
                        <input
                            type="password"
                            value={apiKeyDraft}
                            onChange={lambda e: any  -> None { apiKeyDraft = e.target.value;}}
                            placeholder="sk-..."
                            className="input input--mono"
                        />
                        <div className="settings-actions">
                            <button
                                onClick={handleSaveApiKey}
                                disabled={authBusy}
                                className="app-button app-button--secondary app-button--wide"
                            >
                                {("Saving key..." if authBusy else "Save API Key")}
                            </button>
                            <button
                                onClick={handleDetectAccountSession}
                                disabled={authBusy}
                                className="app-button app-button--ghost app-button--wide"
                            >
                                {("Checking..." if authBusy else "Detect Account Session")}
                            </button>
                        </div>
                    </div>
                    <div className="form-field">
                        <label className="input-label">
                            Auth Status
                        </label>
                        <div className="settings-test tone-muted">
                            <span>
                                {authMessage if authMessage else "No auth status yet"}
                            </span>
                        </div>
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
