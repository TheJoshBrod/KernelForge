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
    anthropic_key = api_keys["anthropic"]
    if api_keys and "anthropic" in api_keys
    else "";
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
        <div className="space-y-6">
            <div className="space-y-2">
                <label className="text-xs font-mono text-[var(--accent)] uppercase tracking-widest block">
                    OpenAI API Key
                </label>
                <div className="relative">
                    <input
                        className="w-full bg-[var(--bg-darkest)] border border-[var(--border)] rounded-sm px-4 py-3 text-sm text-[var(--text-light)] placeholder-[var(--text-muted)] focus:outline-none focus:border-[var(--accent)] focus:ring-1 focus:ring-[var(--accent)]/20 transition-all font-mono shadow-[inset_0_2px_10px_rgba(0,0,0,0.5)] !pr-14"
                        type={("text" if showOpenAI else "password")}
                        value={openai_key}
                        placeholder="sk-..."
                        onChange={lambda e: any  -> None { handleApiKeyChange(
                            "openai", e.target.value
                        );}}
                        spellCheck={False}
                        autoCapitalize="none"
                        autoCorrect="off"
                    />
                    <button
                        type="button"
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)] hover:text-[var(--accent)] transition-colors p-1"
                        onClick={lambda -> None { setShowOpenAI(not showOpenAI);}}
                    >
                        {<EyeOffIcon className="w-5 h-5"/>
                        if showOpenAI
                        else <EyeIcon className="w-5 h-5"/>}
                    </button>
                </div>
            </div>
            <div className="space-y-2">
                <label className="text-xs font-mono text-[var(--accent)] uppercase tracking-widest block">
                    Anthropic API Key
                </label>
                <div className="relative">
                    <input
                        className="w-full bg-[var(--bg-darkest)] border border-[var(--border)] rounded-sm px-4 py-3 text-sm text-[var(--text-light)] placeholder-[var(--text-muted)] focus:outline-none focus:border-[var(--accent)] focus:ring-1 focus:ring-[var(--accent)]/20 transition-all font-mono shadow-[inset_0_2px_10px_rgba(0,0,0,0.5)] !pr-14"
                        type={("text" if showAnthropic else "password")}
                        value={anthropic_key}
                        placeholder="sk-ant-..."
                        onChange={lambda e: any  -> None { handleApiKeyChange(
                            "anthropic", e.target.value
                        );}}
                        spellCheck={False}
                        autoCapitalize="none"
                        autoCorrect="off"
                    />
                    <button
                        type="button"
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)] hover:text-[var(--accent)] transition-colors p-1"
                        onClick={lambda -> None { setShowAnthropic(not showAnthropic);}}
                    >
                        {<EyeOffIcon className="w-5 h-5"/>
                        if showAnthropic
                        else <EyeIcon className="w-5 h-5"/>}
                    </button>
                </div>
            </div>
            <div className="space-y-2">
                <label className="text-xs font-mono text-[var(--accent)] uppercase tracking-widest block">
                    Google API Key
                </label>
                <div className="relative">
                    <input
                        className="w-full bg-[var(--bg-darkest)] border border-[var(--border)] rounded-sm px-4 py-3 text-sm text-[var(--text-light)] placeholder-[var(--text-muted)] focus:outline-none focus:border-[var(--accent)] focus:ring-1 focus:ring-[var(--accent)]/20 transition-all font-mono shadow-[inset_0_2px_10px_rgba(0,0,0,0.5)] !pr-14"
                        type={("text" if showGoogle else "password")}
                        value={google_key}
                        placeholder="AIza..."
                        onChange={lambda e: any  -> None { handleApiKeyChange(
                            "google", e.target.value
                        );}}
                        spellCheck={False}
                        autoCapitalize="none"
                        autoCorrect="off"
                    />
                    <button
                        type="button"
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)] hover:text-[var(--accent)] transition-colors p-1"
                        onClick={lambda -> None { setShowGoogle(not showGoogle);}}
                    >
                        {<EyeOffIcon className="w-5 h-5"/>
                        if showGoogle
                        else <EyeIcon className="w-5 h-5"/>}
                    </button>
                </div>
            </div>
        </div>
    );
}
