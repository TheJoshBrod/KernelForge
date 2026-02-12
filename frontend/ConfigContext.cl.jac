"""Config Context Provider"""

import from react { createContext, useContext, useEffect }
import from "@jac/runtime" { jacSpawn }

glob DEFAULT_CONFIG: dict = {
         "llm_info": {"model": "", "apikey": "", "provider": ""},
         "auth": {
             "mode": "auto",
             "provider": "openai",
             "model": "",
             "api_key_saved": False,
             "account_session_detected": False,
             "last_checked_at": 0
         },
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

def _normalize_cfg(cfg: any) -> dict {
    if not cfg {
        return DEFAULT_CONFIG;
    }
    merged = {
        "llm_info": {
            "model": "",
            "apikey": "",
            "provider": ""
        },
        "auth": {
            "mode": "auto",
            "provider": "openai",
            "model": "",
            "api_key_saved": False,
            "account_session_detected": False,
            "last_checked_at": 0
        },
        "providers": []
    };
    if "llm_info" in cfg and cfg["llm_info"] {
        llm = cfg["llm_info"];
        for key in llm {
            merged["llm_info"][key] = llm[key];
        }
    }
    if "auth" in cfg and cfg["auth"] {
        auth = cfg["auth"];
        for key in auth {
            merged["auth"][key] = auth[key];
        }
    }
    if "providers" in cfg and cfg["providers"] {
        merged["providers"] = cfg["providers"];
    }
    for key in cfg {
        if key != "llm_info" and key != "auth" and key != "providers" {
            merged[key] = cfg[key];
        }
    }
    return merged;
}

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
            if data {
                config = _normalize_cfg(data);
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
                config = _normalize_cfg(newConfig);
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
