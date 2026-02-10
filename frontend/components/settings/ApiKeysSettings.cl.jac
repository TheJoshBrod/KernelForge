"""API Keys Settings Component"""

def:pub ApiKeysSettings(config: any, handleApiKeyChange: any, handleSave: any, saving: bool) -> any {
    return <div className="space-y-4">
        <div className="space-y-4">
            {
                [<div className="space-y-2" key={provider}>
                    <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">{provider}</label>
                    <input
                        type="password"
                        value={config["api_keys"][provider] if "api_keys" in config and provider in config["api_keys"] else ""}
                        onChange={lambda e: any -> None { handleApiKeyChange(provider, e.target.value); }}
                        placeholder={f"Enter {provider} API Key"}
                        className="w-full h-12 bg-zinc-900 border border-zinc-800 hover:border-zinc-700 rounded-lg px-4 text-zinc-200 placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all font-mono text-sm"
                    />
                </div> for provider in ["openai", "gemini", "anthropic"]]
            }
        </div>
        <button
            onClick={handleSave}
            disabled={saving}
            className="w-full h-12 bg-zinc-100 hover:bg-white text-zinc-950 font-bold rounded-lg transition-all shadow-lg hover:shadow-zinc-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
        >
            {("Saving..." if saving else "Save Keys")}
        </button>
    </div>;
}
