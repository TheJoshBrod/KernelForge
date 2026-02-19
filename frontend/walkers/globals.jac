"""Global constants for walkers."""

import from os { path }

glob config_store: dict = {
         "api_keys": {"openai": "", "anthropic": "", "gemini": ""},
         "providers": [],
         "ssh_connections": [],
         "active_ssh_index": -1,
         "selected_gpu_index": -1,
         "selected_gpu_info": None
     },
     config_path: str = "config.json",
     repo_root: str = path.abspath("..");
