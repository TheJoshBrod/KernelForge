"""Global constants for walkers."""

import from os { environ, path }

glob config_store: dict = {
         "api_keys": {"openai": "", "anthropic": "", "google": ""},
         "providers": [],
         "ssh_connections": [],
         "active_ssh_index": -1,
         "selected_gpu_index": -1,
         "selected_gpu_info": None
     },
     config_path: str = (
         str(environ["KFORGE_CONFIG_PATH"])
         if "KFORGE_CONFIG_PATH" in environ and environ["KFORGE_CONFIG_PATH"]
         else path.abspath(path.join("..", "config.json"))
     ),
     repo_root: str = (
         str(environ["KFORGE_REPO_ROOT"])
         if "KFORGE_REPO_ROOT" in environ and environ["KFORGE_REPO_ROOT"]
         else path.abspath("..")
     );
