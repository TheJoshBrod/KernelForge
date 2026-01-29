"""Config Context Provider"""

import from react { createContext, useContext, useState, useEffect }

glob DEFAULT_CONFIG: dict = {
    "llm_info": {"model": "", "apikey": "", "provider": "anthropic"}
};

# Create Context
glob ConfigContext = createContext({
    "config": DEFAULT_CONFIG,
    "setConfig": None,
    "saveConfig": None,
    "loading": True
});

def:pub ConfigProvider(children: any, getConfig: any, saveConfig: any) -> any {
    state = useState(DEFAULT_CONFIG);
    config = state[0];
    setConfig = state[1];

    loading_state = useState(True);
    loading = loading_state[0];
    setLoading = loading_state[1];

    # Load config on mount
    useEffect(
        lambda -> None { async def fetchConfig -> None {
            try {
                data = await getConfig();
                if data and "llm_info" in data {
                    setConfig(data);
                } else {
                    setConfig(DEFAULT_CONFIG);
                }
            } except (error) {
                console.error("Failed to load config:", error);
                setConfig(DEFAULT_CONFIG);
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

def:pub useConfig -> any {
    return useContext(ConfigContext);
}
