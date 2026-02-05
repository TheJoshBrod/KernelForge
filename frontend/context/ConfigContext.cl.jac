"""Config Context Provider"""

import from react { createContext, useContext, useState, useEffect }

# Create Context
glob ConfigContext = createContext();

def:pub ConfigProvider(children: any, getConfig: any, saveConfig: any) -> any {
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
                data = await getConfig();
                # Merge with defaults
                setConfig(
                    lambda prev: any  -> any { return {
                        ** prev,
                        ** data,
                        "llm_info": {** prev["llm_info"], ** data["llm_info"]}
                    }; }
                );
            } except (error) {
                console.error("Failed to load config:", error);
            } finally {
                setLoading(False);
            }
        } fetchConfig();},
        []
    );

    async def save_config_handler(newConfig: any) -> bool {
        try {
            # Calls the passed saveConfig
            res = await saveConfig(newConfig);
            if res {
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
                "saveConfig": save_config_handler,
                "loading": loading
            }}
        >
            {children}
        </ConfigContext.Provider>;
}

def:pub useConfig() -> any {
    return useContext(ConfigContext);
}
