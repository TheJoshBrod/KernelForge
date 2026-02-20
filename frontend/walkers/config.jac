"""Configuration walkers."""

import from json { load, dump }
import from os { path }
import from .globals { config_path, config_store }

def _normalize_provider(provider: str) -> str {
    value = provider.lower().strip() if provider else "";
    if value == "gemini" or value == "google-genai" {
        return "google";
    }
    return value;
}

def _normalize_config(cfg: dict) -> dict {
    raw = cfg if cfg else {};
    merged = {
        "api_keys": {"openai": "", "anthropic": "", "google": ""},
        "providers": raw["providers"]
        if raw and "providers" in raw
        else config_store["providers"],
        "ssh_connections": raw["ssh_connections"]
        if raw and "ssh_connections" in raw
        else config_store["ssh_connections"],
        "active_ssh_index": raw["active_ssh_index"]
        if raw and "active_ssh_index" in raw
        else config_store["active_ssh_index"],
        "selected_gpu_index": raw["selected_gpu_index"]
        if raw and "selected_gpu_index" in raw
        else config_store["selected_gpu_index"],
        "selected_gpu_info": raw["selected_gpu_info"]
        if raw and "selected_gpu_info" in raw
        else config_store["selected_gpu_info"]
    };

    api_keys = {"openai": "", "anthropic": "", "google": ""};
    raw_api_keys = raw["api_keys"] if raw and "api_keys" in raw else {};
    for provider in api_keys {
        if raw_api_keys and provider in raw_api_keys and raw_api_keys[provider] {
            api_keys[provider] = str(raw_api_keys[provider]).strip();
        }
    }

    llm = raw["llm_info"] if raw and "llm_info" in raw else {};
    legacy_provider = _normalize_provider(
        llm["provider"] if llm and "provider" in llm else ""
    );
    legacy_key = str(llm["apikey"]).strip() if llm and "apikey" in llm else "";
    if legacy_provider in api_keys and legacy_key and not api_keys[legacy_provider] {
        api_keys[legacy_provider] = legacy_key;
    }

    merged["api_keys"] = api_keys;
    return merged;
}

walker:pub GetConfig {
    can return_config with entry {
        try {
            abs_path = path.abspath(config_path);
            if not path.exists(abs_path) {
                report _normalize_config(config_store);
                disengage;
            }
            with open(config_path, "r") as f {
                config = load(f);
                report _normalize_config(config);
            }
        } except Exception as e {
            print(f"Error loading config: {e}");
            report _normalize_config(config_store);
        }
    }
}

walker:pub SaveConfig {
    has cfg: dict;

    can update_config with entry {
        try {
            normalized = _normalize_config(self.cfg);
            with open(config_path, "w") as f {
                dump(normalized, f, indent=2);
                global config_store;
                config_store = normalized;
                report True;
            }
        } except Exception as e {
            print(f"Error saving config: {e}");
            report False;
        }
    }
}
