"""API Keys Settings Component"""

import from react { useState }

def:pub ApiKeysSettings(props: any) -> Any {
    config = props.config;
    handleApiKeyChange = props.handleApiKeyChange;
    handleSave = props.handleSave;
    saving = props.saving;

    api_keys = config["api_keys"] if config and "api_keys" in config else {};
    openai_key = api_keys["openai"] if api_keys and "openai" in api_keys else "";
    anthropic_key = api_keys["anthropic"] if api_keys and "anthropic" in api_keys else "";
    google_key = api_keys["google"] if api_keys and "google" in api_keys else "";

    test_state = useState(
        {
            "openai": {"status": "idle", "message": ""},
            "anthropic": {"status": "idle", "message": ""},
            "google": {"status": "idle", "message": ""},
            "all": {"status": "idle", "message": ""}
        }
    );
    testStatus = test_state[0];
    setTestStatus = test_state[1];

    def _tone(status_key: str) -> str {
        if not testStatus or status_key not in testStatus {
            return "tone-muted";
        }
        entry = testStatus[status_key];
        if entry and "status" in entry and entry["status"] == "success" {
            return "tone-success";
        }
        if entry and "status" in entry and entry["status"] == "error" {
            return "tone-error";
        }
        return "tone-muted";
    }

    def _set_provider_status(provider: str, status: str, message: str) -> None {
        next_status = {** testStatus};
        next_status[provider] = {"status": status, "message": message};
        setTestStatus(next_status);
    }

    async def runTest(provider: str, key: str) -> None {
        _set_provider_status(provider, "pending", "Testing...");
        try {
            resp = await __jacSpawn(
                "TestApiKey",
                "",
                {"provider": provider, "api_key": key, "cfg": config}
            );
            data = resp.reports[0] if resp and resp.reports else {};
            success = data["success"] if data and "success" in data else False;
            if success {
                latency = data["latency_ms"] if data and "latency_ms" in data else None;
                msg = "Connection OK";
                if latency is not None {
                    msg = msg + f" ({latency}ms)";
                }
                _set_provider_status(provider, "success", msg);
            } else {
                err = data["error"]
                if data and "error" in data
                else "Connection failed";
                _set_provider_status(provider, "error", err);
            }
        } except Exception as e {
            _set_provider_status(provider, "error", str(e));
        }
    }

    async def runAllTests -> None {
        next_status = {** testStatus};
        next_status["all"] = {"status": "pending", "message": "Testing all keys..."};
        setTestStatus(next_status);
        try {
            resp = await __jacSpawn("TestAllApiKeys", "", {"cfg": config});
            data = resp.reports[0] if resp and resp.reports else {};
            results = data["results"] if data and "results" in data else {};
            out = {** testStatus};
            providers = ["openai", "anthropic", "google"];
            for provider in providers {
                result = results[provider] if results and provider in results else {};
                ok = result["success"] if result and "success" in result else False;
                if ok {
                    latency = result["latency_ms"] if "latency_ms" in result else None;
                    msg = "Connection OK";
                    if latency is not None {
                        msg = msg + f" ({latency}ms)";
                    }
                    out[provider] = {"status": "success", "message": msg};
                } else {
                    err = result["error"] if result and "error" in result else "Connection failed";
                    out[provider] = {"status": "error", "message": err};
                }
            }
            all_ok = data["success"] if data and "success" in data else False;
            out["all"] = {
                "status": ("success" if all_ok else "error"),
                "message": ("All keys valid" if all_ok else "Some key tests failed")
            };
            setTestStatus(out);
        } except Exception as e {
            out = {** testStatus};
            out["all"] = {"status": "error", "message": str(e)};
            setTestStatus(out);
        }
    }

    return (
        <div className="form-stack">
            <div className="form-field">
                <label className="input-label">
                    OpenAI API Key
                </label>
                <input
                    className="input input--mono"
                    type="password"
                    value={openai_key}
                    placeholder="sk-..."
                    onChange={lambda e: any  -> None { handleApiKeyChange("openai", e.target.value);}}
                    spellCheck={False}
                    autoCapitalize="none"
                    autoCorrect="off"
                />
                <div className="settings-actions">
                    <button
                        type="button"
                        className="app-button app-button--secondary app-button--wide"
                        onClick={lambda -> None { runTest("openai", openai_key);}}
                        disabled={testStatus["openai"]["status"] == "pending"}
                    >
                        {("Testing..." if testStatus["openai"]["status"] == "pending" else "Test OpenAI")}
                    </button>
                </div>
                <div className={"settings-test " + _tone("openai")}>
                    {testStatus["openai"]["message"] if testStatus and "openai" in testStatus else ""}
                </div>
            </div>

            <div className="form-field">
                <label className="input-label">
                    Anthropic API Key
                </label>
                <input
                    className="input input--mono"
                    type="password"
                    value={anthropic_key}
                    placeholder="sk-ant-..."
                    onChange={lambda e: any  -> None { handleApiKeyChange("anthropic", e.target.value);}}
                    spellCheck={False}
                    autoCapitalize="none"
                    autoCorrect="off"
                />
                <div className="settings-actions">
                    <button
                        type="button"
                        className="app-button app-button--secondary app-button--wide"
                        onClick={lambda -> None { runTest("anthropic", anthropic_key);}}
                        disabled={testStatus["anthropic"]["status"] == "pending"}
                    >
                        {("Testing..." if testStatus["anthropic"]["status"] == "pending" else "Test Anthropic")}
                    </button>
                </div>
                <div className={"settings-test " + _tone("anthropic")}>
                    {testStatus["anthropic"]["message"] if testStatus and "anthropic" in testStatus else ""}
                </div>
            </div>

            <div className="form-field">
                <label className="input-label">
                    Google API Key
                </label>
                <input
                    className="input input--mono"
                    type="password"
                    value={google_key}
                    placeholder="AIza..."
                    onChange={lambda e: any  -> None { handleApiKeyChange("google", e.target.value);}}
                    spellCheck={False}
                    autoCapitalize="none"
                    autoCorrect="off"
                />
                <div className="settings-actions">
                    <button
                        type="button"
                        className="app-button app-button--secondary app-button--wide"
                        onClick={lambda -> None { runTest("google", google_key);}}
                        disabled={testStatus["google"]["status"] == "pending"}
                    >
                        {("Testing..." if testStatus["google"]["status"] == "pending" else "Test Google")}
                    </button>
                </div>
                <div className={"settings-test " + _tone("google")}>
                    {testStatus["google"]["message"] if testStatus and "google" in testStatus else ""}
                </div>
            </div>

            <div className="settings-actions-block">
                <div className="settings-actions">
                    <button
                        type="button"
                        className="app-button app-button--secondary app-button--wide"
                        onClick={runAllTests}
                        disabled={testStatus["all"]["status"] == "pending"}
                    >
                        {(
                            "Testing..."
                            if testStatus["all"]["status"] == "pending"
                            else "Test All Keys"
                        )}
                    </button>
                    <button
                        type="button"
                        className="app-button app-button--primary app-button--wide"
                        onClick={handleSave}
                        disabled={saving}
                    >
                        {("Saving..." if saving else "Save")}
                    </button>
                </div>
                <div className={"settings-test " + _tone("all")}>
                    {testStatus["all"]["message"] if testStatus and "all" in testStatus else ""}
                </div>
            </div>
        </div>
    );
}
