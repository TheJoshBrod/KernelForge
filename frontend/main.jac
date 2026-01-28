"""Main entry point for frontend."""

import from json { load, dump }
import from pathlib { Path }

glob config_store: dict = {
         "llm_info": {"model": "", "apikey": "", "provider": "anthropic"}
     };

glob config_path: str = Path("config.json");

walker:pub GetConfig {
    can return_config with `root entry {
        try {
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

walker:pub GetProjects {
    can return_projects with `root entry {
        report {"projects": []};
    }
}

cl {
    import from react { useState, useEffect }
    import from sonner { Toaster }
    import from .context.ConfigContext { ConfigProvider }
    import from .components.Home { Home }
    import from .components.Project { Project }
    import from .components.Settings { Settings }

    import "./styles.css";

    def:pub app() -> any {
        has page: str = "/";

        useEffect(lambda -> None { document.title = "CGinS";}, []);

        # Client-side wrappers for walkers
        async def client_get_config() -> dict {
            resp = await root spawn GetConfig();
            return resp.reports[0];
        }

        async def client_save_config(cfg: any) -> bool {
            resp = await root spawn SaveConfig(cfg=cfg);
            return resp.reports[0];
        }

        def navigate(path: str) -> None {
            console.log(f"Navigating to {path}");
            page = path;
            # window.history.pushState({}, "", path);
        }

        parts = page.split("/");
        # path is /project/name -> ["", "project", "name"]
        is_project = (parts[1] == "project") if parts.length > 1 else False;
        project_name = parts[2] if is_project and parts.length > 2 else "";

        return
            <ConfigProvider
                getConfig={client_get_config}
                saveConfig={client_save_config}
            >
                <div
                    className="bg-zinc-950 min-h-screen text-zinc-100 font-sans antialiased"
                >
                    {<Home onNavigate={navigate} />
                    if page == "/" or page == ""
                    else <Settings onNavigate={navigate} />
                    if page == "/settings"
                    else <Project name={project_name} onNavigate={navigate} />
                    if is_project
                    else <div
                        className="flex min-h-screen flex-col items-center justify-center"
                    >
                        <h1>
                            404 Not Found: {page}
                        </h1>
                        <button
                            onClick={lambda -> None { navigate("/");}}
                            className="mt-4 p-2 bg-zinc-800 rounded"
                        >
                            Back to Home
                        </button>
                    </div>}
                    <Toaster richColors={True} theme="dark" />
                </div>
            </ConfigProvider>;
    }
}
