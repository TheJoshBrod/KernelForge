"""Configuration walkers."""

import from json { load, dump }
import from os { path }
import from .globals { config_path, config_store }

walker:pub GetConfig {
    can return_config with `root entry {
        try {
            abs_path = path.abspath(config_path);
            print(f"DEBUG: Loading config from {abs_path}");
            with open(config_path, "r") as f {
                config = load(f);
                report config;
            }
        } except Exception as e {
            print(f"Error loading config: {e}");
            report config_store;
        }
    }
}

walker:pub SaveConfig {
    has cfg: dict;

    can update_config with `root entry {
        try {
            abs_path = path.abspath(config_path);
            with open(config_path, "w") as f {
                dump(self.cfg, f, indent=2);
                global config_store;
                config_store = self.cfg;
                report True;
            }
        } except Exception as e {
            print(f"Error saving config: {e}");
            report False;
        }
    }
}
