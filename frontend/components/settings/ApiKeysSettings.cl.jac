"""API Keys Settings Component"""

import from react { useState }
import from ...components.ui.Icons { EyeIcon, EyeOffIcon }

def:pub ApiKeysSettings(props: any) -> Any {
    config = props.config;
    handleLlmInfoChange = props.handleLlmInfoChange;
    handleSave = props.handleSave;
    saving = props.saving;

    api_keys = config["api_keys"] if config and "api_keys" in config else {};
    openai_key = api_keys["openai"] if api_keys and "openai" in api_keys else "";
    anthropic_key = api_keys["anthropic"] if api_keys and "anthropic" in api_keys else "";
    gemini_key = api_keys["gemini"] if api_keys and "gemini" in api_keys else "";

    # Visibility states
    vis_state = useState({
        "openai": False,
        "anthropic": False,
        "gemini": False
    });
    visibility = vis_state[0];
    setVisibility = vis_state[1];

    def toggleVisibility(provider: str) -> None {
        new_vis = {** visibility};
        new_vis[provider] = not new_vis[provider];
        setVisibility(new_vis);
    }

    test_state = useState({"status": "idle", "message": ""});
    testStatus = test_state[0];
    setTestStatus = test_state[1];

    async def handleTest -> None {
        setTestStatus({"status": "pending", "message": "Testing connection..."});
        try {
            # Test all keys or specific one? For now, let's just test connectivity
            # We'll need to update the backend walker to handle this.
            # For now, sending the config as is.
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
                    OpenAI API Key
                </label>
                <div className="relative">
                    <input
                        className="input input--mono pr-10"
                        type={"text" if visibility["openai"] else "password"}
                        value={openai_key}
                        placeholder="sk-..."
                        onChange={lambda e: any  -> None { handleLlmInfoChange(
                            "openai", e.target.value
                        );}}
                        spellCheck={False}
                        autoCapitalize="none"
                        autoCorrect="off"
                    />
                    <button
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300 transition-colors"
                        onClick={lambda -> None { toggleVisibility("openai"); }}
                    >
                        {
                            <EyeOffIcon className="w-4 h-4" />
                            if visibility["openai"]
                            else
                            <EyeIcon className="w-4 h-4" />
                        }
                    </button>
                </div>
            </div>
            <div className="form-field">
                <label className="input-label">
                    Anthropic API Key
                </label>
                <div className="relative">
                    <input
                        className="input input--mono pr-10"
                        type={"text" if visibility["anthropic"] else "password"}
                        value={anthropic_key}
                        placeholder="sk-ant-..."
                        onChange={lambda e: any  -> None { handleLlmInfoChange(
                            "anthropic", e.target.value
                        );}}
                        spellCheck={False}
                        autoCapitalize="none"
                        autoCorrect="off"
                    />
                    <button
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300 transition-colors"
                        onClick={lambda -> None { toggleVisibility("anthropic"); }}
                    >
                        {
                            <EyeOffIcon className="w-4 h-4" />
                            if visibility["anthropic"]
                            else
                            <EyeIcon className="w-4 h-4" />
                        }
                    </button>
                </div>
            </div>
            <div className="form-field">
                <label className="input-label">
                    Gemini API Key
                </label>
                <div className="relative">
                    <input
                        className="input input--mono pr-10"
                        type={"text" if visibility["gemini"] else "password"}
                        value={gemini_key}
                        placeholder="AIza..."
                        onChange={lambda e: any  -> None { handleLlmInfoChange(
                            "gemini", e.target.value
                        );}}
                        spellCheck={False}
                        autoCapitalize="none"
                        autoCorrect="off"
                    />
                    <button
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300 transition-colors"
                        onClick={lambda -> None { toggleVisibility("gemini"); }}
                    >
                        {
                            <EyeOffIcon className="w-4 h-4" />
                            if visibility["gemini"]
                            else
                            <EyeIcon className="w-4 h-4" />
                        }
                    </button>
                </div>
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
