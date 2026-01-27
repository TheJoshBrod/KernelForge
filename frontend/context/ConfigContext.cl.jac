"""Config Context Provider"""

import from react { createContext, useContext, useState, useEffect }

# Create Context
glob ConfigContext = createContext();

def:pub ConfigProvider(children: any) -> any {
    state = useState(
        {"llm_info": {"model": "", "apikey": "", "provider": "anthropic"}}
    );
    config = state[0];
    setConfig = state[1];

    loading_state = useState(True);
    loading = loading_state[0];
    setLoading = loading_state[1];

    # Load config on mount
    useEffect(
        lambda -> None { async def fetchConfig() -> None {
            try {
                res = await fetch("/api/config");
                if res.ok {
                    data = await res.json();
                    # Merge with defaults
                    setConfig(
                        lambda prev: any  -> any { return {
                            ** prev,
                            ** data,
                            "llm_info": {** prev["llm_info"], ** data["llm_info"]}
                        }; }
                    );
                }
            } except (error) {
                console.error("Failed to load config:", error);
            } finally {
                setLoading(False);
            }
        } fetchConfig();},
        []
    );

    async def saveConfig(newConfig: any) -> bool {
        try {
            res = await fetch(
                "/api/config",
                {
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"},
                    "body": JSON.stringify(newConfig)
                }
            );
            if res.ok {
                setConfig(newConfig);
                return True;
            }
            return False;
        } except (error) {
            console.error("Failed to save config:", error);
            return False;
        }
    }

    return
        <ConfigContext.Provider
            value={{
                "config": config,
                "setConfig": setConfig,
                "saveConfig": saveConfig,
                "loading": loading
            }}
        >
            {children}
        </ConfigContext.Provider>;
}

def:pub useConfig() -> any {
    return useContext(ConfigContext);
}
