"""Config Context Provider"""

import from react { createContext, useContext, useEffect, useState }

glob DEFAULT_CONFIG: dict = {
         "api_keys": {"openai": "", "anthropic": "", "google": ""},
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

    hw_info_state = useState(
        {
            "available": False,
            "gpus": [],
            "source": "local",
            "cuda_available": False,
            "mps_available": False,
            "xpu_available": False,
            "preferred_target": "cpu"
        }
    );
    hwInfo = hw_info_state[0];
    setHwInfo = hw_info_state[1];

    # Load config on mount
    async def fetchConfig -> None {
        try {
            # Run in parallel
            cmd_config = __jacSpawn("GetConfig", "", {});
            cmd_hw = __jacSpawn("DetectHardware", "", {});

            resp_config = await cmd_config;
            resp_hw = await cmd_hw;

            # Process Config
            data = resp_config.reports[0]
            if resp_config and resp_config.reports and resp_config.reports.length > 0
            else None;
            loaded = data if data else {};
            loaded_keys = loaded["api_keys"] if loaded and "api_keys" in loaded else {};
            merged_keys = {** DEFAULT_CONFIG["api_keys"], ** loaded_keys};
            legacy_llm = loaded["llm_info"] if loaded and "llm_info" in loaded else {};
            legacy_provider = (
                legacy_llm["provider"].lower().strip()
                if legacy_llm and "provider" in legacy_llm and legacy_llm["provider"]
                else ""
            );
            if legacy_provider == "gemini" {
                legacy_provider = "google";
            }
            legacy_key = (
                legacy_llm["apikey"]
                if legacy_llm and "apikey" in legacy_llm and legacy_llm["apikey"]
                else ""
            );
            if legacy_provider in merged_keys
            and legacy_key
            and not merged_keys[legacy_provider] {
                merged_keys[legacy_provider] = legacy_key;
            }
            merged = {
                "api_keys": merged_keys,
                "providers": loaded["providers"]
                if loaded and "providers" in loaded
                else DEFAULT_CONFIG["providers"],
                "ssh_connections": loaded["ssh_connections"]
                if loaded and "ssh_connections" in loaded
                else DEFAULT_CONFIG["ssh_connections"],
                "active_ssh_index": loaded["active_ssh_index"]
                if loaded and "active_ssh_index" in loaded
                else DEFAULT_CONFIG["active_ssh_index"],
                "selected_gpu_index": loaded["selected_gpu_index"]
                if loaded and "selected_gpu_index" in loaded
                else DEFAULT_CONFIG["selected_gpu_index"],
                "selected_gpu_info": loaded["selected_gpu_info"]
                if loaded and "selected_gpu_info" in loaded
                else DEFAULT_CONFIG["selected_gpu_info"]
            };

            # Normalize SSH config
            conns = merged["ssh_connections"] if "ssh_connections" in merged else [];
            active_idx = merged["active_ssh_index"]
            if "active_ssh_index" in merged
            else -1;
            if active_idx < -1 or active_idx >= conns.length {
                merged["active_ssh_index"] = -1;
            }

            # Process Hardware Info
            hw_data = resp_hw.reports[0]
            if resp_hw and resp_hw.reports and resp_hw.reports.length > 0
            else {};
            # Normalize selected GPU index
            selected_idx = merged["selected_gpu_index"]
            if "selected_gpu_index" in merged
            else -1;
            if selected_idx < -1 {
                merged["selected_gpu_index"] = -1;
                merged["selected_gpu_info"] = None;
            }
            if (
                merged["active_ssh_index"] == -1
                and merged["selected_gpu_index"] == -1
                and hw_data and "gpus" in hw_data and hw_data["gpus"]
            ) {
                merged["selected_gpu_index"] = 0;
                merged["selected_gpu_info"] = hw_data["gpus"][0];
            }
            setConfigState(merged);
            setHwInfo(hw_data);
        } except Exception as e {
            console.error("Failed to load config/hardware:", e);
            err_str = e.toString();
            if err_str.includes("Invalid anchor id")
            or err_str.includes("500")
            or err_str.includes("Internal Server Error") {
                console.warn("Detected stale session.");
                # Retry Loop Protection
                stored_retries = localStorage.getItem("session_retries");
                retry_count = parseInt(stored_retries) if stored_retries else 0;
                if retry_count >= 3 {
                    console.error("Critical: Max session retries exceeded.");
                    alert(
                        "Session Error: Please manually clear your browser cookies and refresh the page."
                    );
                    # Stop here to prevent infinite loop
                    setLoading(False);
                    return;
                }
                console.warn(
                    "Resetting session (Attempt " + (retry_count + 1).toString() + "/3)..."
                );
                localStorage.clear();
                sessionStorage.clear();
                # Persist retry count after clear
                localStorage.setItem("session_retries", (retry_count + 1).toString());
                window.location.reload();
            }
            setConfigState(DEFAULT_CONFIG);
        } finally {
            setLoading(False);
        }
    }

    useEffect(lambda -> None { fetchConfig();}, []);

    async def save_config_handler(newConfig: dict) -> bool {
        try {
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
                "loading": loading,
                "hwInfo": hwInfo
            }}
        >
            {children}
        </ConfigContext.Provider>;
}

def:pub useConfig -> Any {
    return useContext(ConfigContext);
}
# Create Context
