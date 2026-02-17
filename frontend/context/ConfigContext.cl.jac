"""Config Context Provider"""

import from react { createContext, useContext, useEffect, useState }

glob DEFAULT_CONFIG: dict = {
         "llm_info": {"model": "", "apikey": "", "provider": ""},
         "providers": [],
         "ssh_connections": [],
         "active_ssh_index": -1,
         "selected_gpu_index": -1,
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
            resp = await __jacSpawn("GetConfig", "", {});
            data = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else None;
            loaded = data if data else {};
            loaded_llm = loaded["llm_info"] if loaded and "llm_info" in loaded else {};
            merged_llm = {** DEFAULT_CONFIG["llm_info"], ** loaded_llm};
            merged = {** DEFAULT_CONFIG, ** loaded};
            merged["llm_info"] = merged_llm;

            # Normalize SSH config to avoid out-of-range active index
            conns = merged["ssh_connections"] if "ssh_connections" in merged else [];
            active_idx = merged["active_ssh_index"]
            if "active_ssh_index" in merged
            else -1;
            if active_idx < -1 or active_idx >= conns.length {
                merged["active_ssh_index"] = -1;
            }

            # Normalize selected GPU index
            selected_idx = merged["selected_gpu_index"]
            if "selected_gpu_index" in merged
            else -1;
            if selected_idx < -1 {
                merged["selected_gpu_index"] = -1;
            }

            setConfigState(merged);
        } except Exception as e {
            console.error("Failed to load config:", e);
            setConfigState(DEFAULT_CONFIG);
        } finally {
            setLoading(False);
        }
    }

    useEffect(lambda -> None { fetchConfig();}, []);

    async def save_config_handler(newConfig: dict) -> bool {
        try {
            # Calls the passed saveConfig
            resp = await __jacSpawn("SaveConfig", "", {"cfg": newConfig});
            res = resp.reports[0]
            if resp and resp.reports and resp.reports.length > 0
            else False;
            if res {
                setConfigState(newConfig);
                return True;
            }
            return False;
        } except Exception as e {
            console.error("Failed to save config:", e);
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
