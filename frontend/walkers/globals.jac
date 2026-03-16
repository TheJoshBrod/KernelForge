"""Global constants for walkers."""

import from os { environ, path }
import from os.path { isdir }
import from os.path { expanduser }

glob config_store: dict = {
         "api_keys": {"openai": "", "anthropic": "", "google": ""},
         "providers": [],
         "ssh_connections": [],
         "active_ssh_index": -1,
         "selected_gpu_index": -1,
         "selected_gpu_info": None
     },
     repo_root: str = "";

def _derive_repo_root() -> str {
    if "KFORGE_REPO_ROOT" in environ and environ["KFORGE_REPO_ROOT"] {
        return path.abspath(expanduser(environ["KFORGE_REPO_ROOT"]));
    }
    return path.abspath("..");
}

def _derive_data_root() -> str {
    if "KFORGE_DATA_DIR" in environ and environ["KFORGE_DATA_DIR"] {
        return path.abspath(expanduser(environ["KFORGE_DATA_DIR"]));
    }
    legacy_private = path.join(repo_root, "kernels", "private");
    legacy = path.join(repo_root, "kernels");
    if isdir(legacy_private) {
        return legacy_private;
    }
    return legacy;
}

repo_root = _derive_repo_root();
data_root = _derive_data_root();
config_path = environ["KFORGE_CONFIG_PATH"]
            if "KFORGE_CONFIG_PATH" in environ and environ["KFORGE_CONFIG_PATH"] else path.join(data_root, "config.json");
