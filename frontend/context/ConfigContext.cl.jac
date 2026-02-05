"""Config Context Provider"""

import from react { createContext, useContext, useEffect }
import from "@jac/runtime" { jacSpawn }

glob DEFAULT_CONFIG: dict = {
         "llm_info": {"model": "", "apikey": "", "provider": ""},
         "providers": []
     },
     ConfigContext = createContext(
         {
             "config": DEFAULT_CONFIG,
             "setConfig": None,
             "saveConfig": None,
             "loading": True
         }
     );

def:pub ConfigProvider(children: any) -> any {
    has config: dict = DEFAULT_CONFIG,
        loading: bool = True;

    # Load config on mount
    async def fetchConfig -> None {
        try {
            resp = await jacSpawn("GetConfig", "", {});
            data = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else None;
            if data and "llm_info" in data {
                config = data;
            } else {
                config = DEFAULT_CONFIG;
            }
        } except (error) {
            console.error("Failed to load config:", error);
            config = DEFAULT_CONFIG;
        } finally {
            loading = False;
        }
    }

    useEffect(lambda -> None { fetchConfig();}, []);

    async def save_config_handler(newConfig: any) -> bool {
        try {
            # Calls the passed saveConfig
            resp = await jacSpawn("SaveConfig", "", {"cfg": newConfig});
            res = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else False;
            if res {
                config = newConfig;
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
                "setConfig": lambda newConfig: any  -> None { config = newConfig;},
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
# Create Context
