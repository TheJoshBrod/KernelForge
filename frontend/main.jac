"""Main entry point for frontend."""

import from json { load, dump }
import from pathlib { Path }
import from base64 { b64decode }
import from os { path, makedirs, listdir }
import from datetime { datetime }

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

walker:pub CreateProject {
    has projectName: str;
    has code: str;
    has weightsFileB64: str;
    has validationSetPath: str;

    can create with `root entry {
        try {
            # Define projects directory
            projects_dir = "projects";
            if not path.exists(projects_dir) {
                makedirs(projects_dir);
            }

            project_path = path.join(projects_dir, self.projectName);

            if path.exists(project_path) {
                report {"error": "Project already exists", "success": False};
                disengage;
            }

            makedirs(project_path);

            # Save model.py
            if self.code {
                with open(path.join(project_path, "model.py"), "w") as f {
                    f.write(self.code);
                }
            }

            # Save weights.pt
            if self.weightsFileB64 {
                weights_data = b64decode(self.weightsFileB64);
                with open(path.join(project_path, "weights.pt"), "wb") as f {
                    f.write(weights_data);
                }
            }

            # Save config.json
            config = {
                "validation_set": self.validationSetPath
                if self.validationSetPath
                else None,
                "created_at": datetime.now().isoformat()
            };
            with open(path.join(project_path, "config.json"), "w") as f {
                dump(config, f, indent=2);
            }

            report {"success": True, "name": self.projectName};
        } except Exception as e {
            print(f"Error creating project: {e}");
            report {"error": "Failed to create project", "success": False};
        }
    }
}

walker:pub GetProjects {
    can return_projects with `root entry {
        try {
            projects_dir = "projects";
            if not path.exists(projects_dir) {
                report {"projects": []};
                disengage;
            }

            projects = [];
            for name in listdir(projects_dir) {
                full_path = path.join(projects_dir, name);
                if path.isdir(full_path) {
                    projects.append({"name": name});
                }
            }
            report {"projects": projects};
        } except Exception as e {
            print(f"Error listing projects: {e}");
            report {"projects": []};
        }
    }
}

cl {
    import from react { useState, useEffect }
    import from sonner { Toaster }
    import from .context.ConfigContext { ConfigProvider }
    import from .components.Home { Home }
    import from .components.Project { Project }
    import from .components.Settings { Settings }
    import from .components.NewProject { NewProject }
    import from .components.OpDetails { OpDetails }

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

        async def client_create_project(
            projectName: str, code: str, weightsFileB64: str, validationSetPath: str
        ) -> dict {
            resp = await root spawn CreateProject(
                projectName=projectName,
                code=code,
                weightsFileB64=weightsFileB64,
                validationSetPath=validationSetPath
            );
            return resp.reports[0] if resp.reports else {};
        }

        async def client_get_projects() -> dict {
            resp = await root spawn GetProjects();
            return resp.reports[0] if resp.reports else {"projects": []};
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

        is_op_details = is_project and parts.length > 3;
        op_name = parts[3] if is_op_details else "";

        return
            <ConfigProvider
                getConfig={client_get_config}
                saveConfig={client_save_config}
            >
                <div
                    className="bg-zinc-950 min-h-screen text-zinc-100 font-sans antialiased"
                >
                    {<Home
                        onNavigate={navigate}
                        getProjectsWalker={client_get_projects}
                    />
                    if page == "/" or page == ""
                    else <Settings onNavigate={navigate} />
                    if page == "/settings"
                    else <NewProject
                        onNavigate={navigate}
                        createProjectWalker={client_create_project}
                    />
                    if page == "/new-project"
                    else <OpDetails
                        projectName={project_name}
                        opName={op_name}
                        onNavigate={navigate}
                    />
                    if is_op_details
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
