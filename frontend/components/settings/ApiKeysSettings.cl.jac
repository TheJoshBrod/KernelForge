"""API Keys Settings Component"""

import from react { useState }

def:pub ApiKeysSettings(props: any) -> Any {
    config = props.config;
    handleLlmInfoChange = props.handleLlmInfoChange;
    handleSave = props.handleSave;
    saving = props.saving;

    llm = config["llm_info"] if config and "llm_info" in config else {};
    provider = llm["provider"] if llm and "provider" in llm else "";
    model = llm["model"] if llm and "model" in llm else "";
    apikey = llm["apikey"] if llm and "apikey" in llm else "";

    test_state = useState({"status": "idle", "message": ""});
    testStatus = test_state[0];
    setTestStatus = test_state[1];

    async def handleTest -> None {
        setTestStatus({"status": "pending", "message": "Testing connection..."});
        try {
            resp = await __jacSpawn("TestLLMConnection", "", {"cfg": config});
            data = resp.reports[0] if resp and resp.reports else {};
            success = data["success"] if data and "success" in data else False;
            if success {
                latency = data["latency_ms"] if "latency_ms" in data else None;
                msg = "Connection OK";
                if latency is not None {
                    msg = msg + f" ({latency}ms)";
                }
                setTestStatus({"status": "success", "message": msg});
            } else {
                err = data["error"]
                if data and "error" in data
                else "Connection failed";
                setTestStatus({"status": "error", "message": err});
            }
        } except Exception as e {
            setTestStatus({"status": "error", "message": str(e)});
        }
    }

    tone = "tone-muted";
    if testStatus["status"] == "success" {
        tone = "tone-success";
    } elif testStatus["status"] == "error" {
        tone = "tone-error";
    }

    return (
        <div className="form-stack">
            <div className="form-field">
                <label className="input-label">
                    Provider
                </label>
                <input
                    className="input input--mono"
                    type="text"
                    value={provider}
                    placeholder="openai | anthropic | gemini"
                    onChange={lambda e: any  -> None { handleLlmInfoChange(
                        "provider", e.target.value
                    );}}
                    spellCheck={False}
                    autoCapitalize="none"
                    autoCorrect="off"
                />
            </div>
            <div className="form-field">
                <label className="input-label">
                    Model
                </label>
                <input
                    className="input input--mono"
                    type="text"
                    value={model}
                    placeholder="gpt-4.1-mini | claude-3-5-sonnet-latest | gemini-2.0-flash"
                    onChange={lambda e: any  -> None { handleLlmInfoChange(
                        "model", e.target.value
                    );}}
                    spellCheck={False}
                    autoCapitalize="none"
                    autoCorrect="off"
                />
            </div>
            <div className="form-field">
                <label className="input-label">
                    API Key
                </label>
                <input
                    className="input input--mono"
                    type="password"
                    value={apikey}
                    placeholder="sk-..."
                    onChange={lambda e: any  -> None { handleLlmInfoChange(
                        "apikey", e.target.value
                    );}}
                    spellCheck={False}
                    autoCapitalize="none"
                    autoCorrect="off"
                />
            </div>
            <div className="settings-actions-block">
                <div className="settings-actions">
                    <button
                        type="button"
                        className="app-button app-button--secondary app-button--wide"
                        onClick={handleTest}
                        disabled={testStatus["status"] == "pending"}
                    >
                        {(
                            "Testing..."
                            if testStatus["status"] == "pending"
                            else "Test"
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
                <div className={"settings-test " + tone}>
                    {testStatus["message"] if "message" in testStatus else ""}
                </div>
            </div>
        </div>
    );
}
