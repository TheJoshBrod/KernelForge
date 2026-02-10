"""Global constants for walkers."""

import from os { path }

glob config_store: dict = {
         "llm_info": {"model": "", "apikey": "", "provider": ""},
         "providers": []
     },
     config_path: str = "config.json",
     repo_root: str = path.abspath("..");
