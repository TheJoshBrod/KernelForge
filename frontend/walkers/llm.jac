"""LLM related walker:pubs and utilities."""

import os;
import json;
import re;
import sys;
import time;
import openai;
import anthropic;
import from google { genai }
import requests;

# Helper functions definition
def fetch_openai_models(api_key: str) -> list {
    try {
        client = openai.OpenAI(api_key=api_key);
        models = client.models.list();

        chat_models = [];
        if models and hasattr(models, 'data') {
            for m in models.data {
                mid = m.id.lower();
                if not (
                    mid.startswith("gpt-")
                    or mid.startswith("o1-")
                    or mid.startswith("o3-")
                ) {
                    continue;
                }

                if "audio" in mid or "realtime" in mid or "voice" in mid {
                    continue;
                }
                if "tts" in mid or "transcribe" in mid or "search-api" in mid {
                    continue;
                }
                if "image" in mid or "dall-e" in mid {
                    continue;
                }

                chat_models.append(
                    {
                        "id": m.id,
                        "created": m.created,
                        "provider": "openai",
                        "label": m.id
                    }
                );
            }
        }
        return chat_models;
    } except Exception as e {
        return [];
    }
}

def fetch_anthropic_models(api_key: str) -> list {
    try {
        client = anthropic.Anthropic(api_key=api_key);
        models_page = client.models.list();

        ant_models = [];
        for m in models_page.data {
            ant_models.append(
                {
                    "id": m.id,
                    "created": 0,
                    "provider": "anthropic",
                    "label": m.display_name if hasattr(m, 'display_name') else m.id
                }
            );
        }
        return ant_models;
    } except Exception as e {
        return [];
    }
}

def fetch_gemini_models(api_key: str) -> list {
    try {
        client = genai.Client(api_key=api_key);
        all_models = list(client.models.list());

        gem_models = [];
        for m in all_models {
            model_name = getattr(m, 'name', '');
            if 'gemini' in model_name.lower() {
                methods = getattr(m, 'supported_generation_methods', []);
                if methods and 'generateContent' not in methods {
                    continue;
                }
                mid = model_name.lower();
                if "embedding" in mid {
                    continue;
                }
                if "image" in mid {
                    continue;
                }
                if "vision" in mid and "gemini-pro-vision" not in mid {
                    continue;
                }
                if "robotics" in mid {
                    continue;
                }
                if "computer-use" in mid {
                    continue;
                }
                if "tts" in mid or "audio" in mid {
                    continue;
                }
                clean_id = model_name.replace("models/", "");
                gem_models.append(
                    {
                        "id": clean_id,
                        "created": 0,
                        "provider": "gemini",
                        "label": getattr(m, 'display_name', clean_id)
                    }
                );
            }
        }
        return gem_models;
    } except Exception as e {
        return [];
    }
}

def get_gemini_sort_key(model: dict) -> tuple {
    model_id = model.get("id", "");
    version_match = re.search(r'(\d+\.?\d*)', model_id);
    version = 0.0;
    try {
        version = float(version_match.group(1)) if version_match else 0.0;
    } except ValueError {
        version = 0.0;
    }

    if 'exp' in model_id and version == 0.0 {
        version = 2.0;
    }

    date_match = re.findall(r'(\d{2}-?\d{2,4})', model_id);

    date_val = 0;
    if date_match {
        date_str = date_match[-1].replace('-', '');
        if len(date_str) == 4 {
            if "1206" in date_str {
                date_val = int(f"2024{date_str}");
            } elif "0924" in date_str {
                date_val = int(f"2024{date_str}");
            } else {
                if date_str.startswith("20") {
                    date_val = int(date_str) * 100;
                } else {
                    date_val = int(f"2025{date_str}");
                }
            }
        } elif len(date_str) == 6 {
            month = date_str[:2];
            year = date_str[2:];
            date_val = int(f"{year}{month}");
        }
    }

    STABLE_VAL = 50000000;
    LATEST_VAL = 90000000;

    if date_val == 0 {
        date_val = STABLE_VAL;
    }

    if 'latest' in model_id {
        date_val = LATEST_VAL;
        if version == 0.0 {
            version = 100.0;
        }
    }

    return (date_val, version);
}

def test_connection_helper(provider: str, apikey: str) -> dict {
    start_time = time.time();
    success = False;
    error_msg = "";

    try {
        if provider == "openai" {
            headers = {"Authorization": f"Bearer {apikey}"};
            res = requests.get("https://api.openai.com/v1/models", headers=headers);
            success = res.status_code == 200;
            if not success {
                error_msg = res.text;
            }
        } elif provider == "anthropic" {
            headers = {"x-api-key": apikey, "anthropic-version": "2023-06-01"};
            res = requests.get("https://api.anthropic.com/v1/models", headers=headers);
            success = res.status_code == 200;
            if not success {
                error_msg = res.text;
            }
        } elif provider == "gemini" {
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={apikey}";
            res = requests.get(url);
            success = res.status_code == 200;
            if not success {
                error_msg = res.text;
            }
        }
    } except Exception as e {
        error_msg = str(e);
        success = False;
    }

    duration_ms = int((time.time() - start_time) * 1000);

    return {
        "success": success,
        "error": error_msg,
        "latency_ms": duration_ms,
        "provider": provider
    };
}

walker:pub GetOpenLLMModels {
    has cfg: dict = {};

    can run with entry {
        import from .globals { config_store, config_path }

        cfg = self.cfg;
        # Check if provided config has useful keys
        has_keys = False;
        if cfg and "api_keys" in cfg {
            for (k, v) in cfg["api_keys"].items() {
                if v {
                    has_keys = True;
                    break;
                }
            }
        }

        if not has_keys {
            try {
                with open(config_path, "r") as f {
                    loaded = json.load(f);
                    if loaded and "api_keys" in loaded {
                        cfg = loaded;
                    }
                }
            } except Exception as e {
                print(f"Error loading config: {e}");
            }
        }

        api_keys = cfg["api_keys"] if cfg and "api_keys" in cfg else {};

        all_models = [];

        if "openai" in api_keys and api_keys["openai"] {
            all_models.extend(fetch_openai_models(api_keys["openai"]));
        }
        if "anthropic" in api_keys and api_keys["anthropic"] {
            all_models.extend(fetch_anthropic_models(api_keys["anthropic"]));
        }
        if "gemini" in api_keys and api_keys["gemini"] {
            all_models.extend(fetch_gemini_models(api_keys["gemini"]));
        }

        # Sorting
        openai_models = [
            m
            for m in all_models
            if m.get("provider") == "openai"
        ];
        anthropic_models = [
            m
            for m in all_models
            if m.get("provider") == "anthropic"
        ];
        gemini_models = [
            m
            for m in all_models
            if m.get("provider") == "gemini"
        ];
        others = [
            m
            for m in all_models
            if m.get("provider") not in ["openai", "anthropic", "gemini"]
        ];

        openai_models.sort(key=lambda x : x.get("created", 0), reverse=True);
        anthropic_models.sort(key=lambda x : x.get("id", ""), reverse=True);
        gemini_models.sort(key=get_gemini_sort_key, reverse=True);

        sorted_models = openai_models + anthropic_models + gemini_models + others;

        report {"success": True, "models": sorted_models};
    }
}

walker:pub TestLLMConnection {
    has cfg: dict = {};

    can run with entry {
        import from .globals { config_store, config_path }

        cfg = self.cfg;
        if not cfg {
            try {
                with open(config_path, "r") as f {
                    cfg = json.load(f);
                }
            } except Exception as e {
                print(f"Error loading config: {e}");
                cfg = config_store;
            }
        }

        api_keys = cfg["api_keys"] if cfg and "api_keys" in cfg else {};
        results = [];

        # Define default test models
        test_models = {
            "openai": "gpt-5-nano",
            "anthropic": "claude-3-haiku-latest",
            "gemini": "gemini-2.5-flash-lite"
        };

        has_keys = False;
        for provider in ["openai", "anthropic", "gemini"] {
            key = api_keys[provider] if provider in api_keys else "";
            if key {
                has_keys = True;
                model = test_models[provider];
                res = test_connection_helper(provider, key);
                results.append(res);
            }
        }

        if not has_keys {
            report {"success": False, "error": "No API keys configured."};
        } else {
            # Aggregate results
            all_success = True;
            errors = [];
            latencies = [];
            for r in results {
                if not r["success"] {
                    all_success = False;
                    errors.append(
                        f"{r.get('provider', 'Unknown')}: {r.get('error', 'Failed')}"
                    );
                } else {
                    latencies.append(r.get("latency_ms", 0));
                }
            }
            if all_success {
                avg_latency = sum(latencies) / len(latencies) if latencies else 0;
                report {
                    "success": True,
                    "latency_ms": int(avg_latency),
                    "message": "All keys valid."
                };
            } else {
                report {"success": False, "error": "; ".join(errors)};
            }
        }
    }
}
