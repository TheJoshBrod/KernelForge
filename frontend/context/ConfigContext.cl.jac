"""Config Context Provider"""

import from react { createContext, useContext, useEffect, useState }
import from "@jac/runtime" { jacSpawn }

glob DEFAULT_CONFIG: dict = {
         "llm_info": {"model": "", "apikey": "", "provider": ""},
         "providers": [],
         "selected_gpu_info": None
     },
     ConfigContext = createContext(
         {
             "config": DEFAULT_CONFIG,
             "setConfig": None,
             "saveConfig": None,
             "loading": True
         }
     );

def:pub ConfigProvider(children: any) -> Any {
    # Use React state so context updates trigger re-renders
    config_state = useState(DEFAULT_CONFIG);
    config = config_state[0];
    setConfigState = config_state[1];

    loading_state = useState(True);
    loading = loading_state[0];
    setLoading = loading_state[1];

    # Load config on mount
    async def fetchConfig -> None {
        try {
            resp = await jacSpawn("GetConfig", "", {});
            data = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else None;
            if data and "llm_info" in data {
                # Normalize SSH config to avoid out-of-range active index
                if "ssh_connections" in data {
                    conns = data["ssh_connections"];
                    if "active_ssh_index" in data {
                        idx = data["active_ssh_index"];
                        if idx < -1 or idx >= conns.length {
                            data["active_ssh_index"] = -1;
                        }
                    }
                }
                setConfigState(data);
            } else {
                setConfigState(DEFAULT_CONFIG);
            }
        } except (error) {
            console.error("Failed to load config:", error);
            setConfigState(DEFAULT_CONFIG);
        } finally {
            setLoading(False);
        }
    }

    useEffect(lambda -> None { fetchConfig();}, []);

    async def save_config_handler(newConfig: dict) -> bool {
        try {
            # Calls the passed saveConfig
            resp = await jacSpawn("SaveConfig", "", {"cfg": newConfig});
            res = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else False;
            if res {
                setConfigState(newConfig);
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
                "setConfig": lambda newConfig: dict  -> None { setConfigState(
                    newConfig
                );},
                "saveConfig": save_config_handler,
                "loading": loading
            }}
        >
            {children}
        </ConfigContext.Provider>;
}

def:pub useConfig {
    return useContext(ConfigContext);
}
# Create Context
