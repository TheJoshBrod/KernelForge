"""API Keys Settings Component"""

import from react { useState }
import from ...components.ui.Icons { EyeIcon, EyeOffIcon }

def:pub ApiKeysSettings(props: any) -> Any {
    config = props.config;
    handleApiKeyChange = props.handleApiKeyChange;
    handleSave = props.handleSave;
    saving = props.saving;

    api_keys = config["api_keys"] if config and "api_keys" in config else {};
    openai_key = api_keys["openai"] if api_keys and "openai" in api_keys else "";
    anthropic_key = api_keys["anthropic"] if api_keys and "anthropic" in api_keys else "";
    google_key = api_keys["google"] if api_keys and "google" in api_keys else "";

    # Individual visibility states
    show_openai_state = useState(False);
    showOpenAI = show_openai_state[0];
    setShowOpenAI = show_openai_state[1];

    show_anthropic_state = useState(False);
    showAnthropic = show_anthropic_state[0];
    setShowAnthropic = show_anthropic_state[1];

    show_google_state = useState(False);
    showGoogle = show_google_state[0];
    setShowGoogle = show_google_state[1];

    return (
        <div className="form-stack">
            <div className="form-field">
                <label className="input-label">
                    OpenAI API Key
                </label>
                <div className="relative">
                    <input
                        className="input input--mono !pr-14"
                        type={("text" if showOpenAI else "password")}
                        value={openai_key}
                        placeholder="sk-..."
                        onChange={lambda e: any  -> None { handleApiKeyChange("openai", e.target.value);}}
                        spellCheck={False}
                        autoCapitalize="none"
                        autoCorrect="off"
                    />
                    <button
                        type="button"
                        className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300 transition-colors p-1"
                        onClick={lambda -> None { setShowOpenAI(not showOpenAI); }}
                    >
                        {<EyeOffIcon className="w-4 h-4" /> if showOpenAI else <EyeIcon className="w-4 h-4" />}
                    </button>
                </div>
            </div>

            <div className="form-field">
                <label className="input-label">
                    Anthropic API Key
                </label>
                <div className="relative">
                    <input
                        className="input input--mono !pr-14"
                        type={("text" if showAnthropic else "password")}
                        value={anthropic_key}
                        placeholder="sk-ant-..."
                        onChange={lambda e: any  -> None { handleApiKeyChange("anthropic", e.target.value);}}
                        spellCheck={False}
                        autoCapitalize="none"
                        autoCorrect="off"
                    />
                     <button
                        type="button"
                        className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300 transition-colors p-1"
                        onClick={lambda -> None { setShowAnthropic(not showAnthropic); }}
                    >
                        {<EyeOffIcon className="w-4 h-4" /> if showAnthropic else <EyeIcon className="w-4 h-4" />}
                    </button>
                </div>
            </div>

            <div className="form-field">
                <label className="input-label">
                    Google API Key
                </label>
                <div className="relative">
                    <input
                        className="input input--mono !pr-14"
                        type={("text" if showGoogle else "password")}
                        value={google_key}
                        placeholder="AIza..."
                        onChange={lambda e: any  -> None { handleApiKeyChange("google", e.target.value);}}
                        spellCheck={False}
                        autoCapitalize="none"
                        autoCorrect="off"
                    />
                     <button
                        type="button"
                        className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300 transition-colors p-1"
                        onClick={lambda -> None { setShowGoogle(not showGoogle); }}
                    >
                        {<EyeOffIcon className="w-4 h-4" /> if showGoogle else <EyeIcon className="w-4 h-4" />}
                    </button>
                </div>
            </div>


        </div>
    );
}
