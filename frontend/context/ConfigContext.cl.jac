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

    def normalizeSelectedGpu(cfg: dict, hw: dict) -> dict {
        merged_cfg = {** cfg};
        gpus = hw["gpus"] if hw and "gpus" in hw and hw["gpus"] else [];
        selected_idx = merged_cfg["selected_gpu_index"]
        if merged_cfg and "selected_gpu_index" in merged_cfg
        else -1;
        selected_info = merged_cfg["selected_gpu_info"]
        if merged_cfg and "selected_gpu_info" in merged_cfg
        else None;

        if selected_idx != -1 and selected_idx < gpus.length {
            merged_cfg["selected_gpu_info"] = gpus[selected_idx];
            return merged_cfg;
        }

        if not selected_info or gpus.length == 0 {
            if selected_idx != -1 and selected_idx >= gpus.length {
                merged_cfg["selected_gpu_index"] = -1;
                merged_cfg["selected_gpu_info"] = None;
            }
            return merged_cfg;
        }

        match_idx = -1;
        selected_uuid = (
            str(selected_info["uuid"])
            if "uuid" in selected_info and selected_info["uuid"]
            else ""
        );
        selected_name = (
            str(selected_info["name"])
            if "name" in selected_info and selected_info["name"]
            else ""
        );
        selected_conn = (
            str(selected_info["connection"])
            if "connection" in selected_info and selected_info["connection"]
            else "local"
        );
        selected_device = (
            str(selected_info["device_id"])
            if "device_id" in selected_info and selected_info["device_id"]
            else ""
        );

        idx = 0;
        for gpu in gpus {
            gpu_uuid = str(gpu["uuid"]) if "uuid" in gpu and gpu["uuid"] else "";
            gpu_name = str(gpu["name"]) if "name" in gpu and gpu["name"] else "";
            gpu_conn = (
                str(gpu["connection"])
                if "connection" in gpu and gpu["connection"]
                else "local"
            );
            gpu_device = (
                str(gpu["device_id"])
                if "device_id" in gpu and gpu["device_id"]
                else ""
            );
            if selected_uuid and gpu_uuid and selected_uuid == gpu_uuid {
                match_idx = idx;
                break;
            }
            if gpu_name == selected_name and gpu_conn == selected_conn {
                if not selected_device or not gpu_device or gpu_device == selected_device {
                    match_idx = idx;
                    break;
                }
            }
            idx += 1;
        }

        if match_idx != -1 {
            merged_cfg["selected_gpu_index"] = match_idx;
            merged_cfg["selected_gpu_info"] = gpus[match_idx];
        } else {
            merged_cfg["selected_gpu_index"] = -1;
            merged_cfg["selected_gpu_info"] = None;
        }
        return merged_cfg;
    }

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

            # Normalize selected GPU index
            selected_idx = merged["selected_gpu_index"]
            if "selected_gpu_index" in merged
            else -1;
            if selected_idx < -1 {
                merged["selected_gpu_index"] = -1;
            }
            # Process Hardware Info
            hw_data = resp_hw.reports[0]
            if resp_hw and resp_hw.reports and resp_hw.reports.length > 0
            else {};
            normalized_config = normalizeSelectedGpu(merged, hw_data);
            setConfigState(normalized_config);
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
