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
    has weightsFileB64: str = "";
    has validationSetPath: str = "";

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

walker:pub GetOptimizationData {
    has projectName: str;
    has opName: str;

    can return_data with `root entry {
        try {
            # HARDCODED PATH AS REQUESTED BY USER FOR DEBUGGING
            target_suffix = "kernels/optimized/NVIDIA_GeForce_GTX_1660_Ti_egr6dqhOEU/torch_nn_functional_linear/nodes";
            
            # Try finding it relative to CWD (either root or frontend/)
            nodes_dir = target_suffix;
            if not path.exists(nodes_dir) {
                # Try going up one level if we are in frontend/
                nodes_dir = path.join("..", target_suffix);
            }
            
            if not path.exists(nodes_dir) {
                 print(f"HARDCODED PATH NOT FOUND. Tried: {target_suffix} and ../{target_suffix}");
            }

            print(f"Reading from: {nodes_dir}");

            nodes_data = [];
            if path.exists(nodes_dir) {
                files = listdir(nodes_dir);
                for file_name in files {
                    if file_name.endswith(".json") {
                        file_path = path.join(nodes_dir, file_name);
                        with open(file_path, "r") as f {
                            node_data_item = load(f);
                            nodes_data.append(node_data_item);
                        }
                    }
                }
            }
            
            report {"nodes": nodes_data};

        } except Exception as e {
            print(f"Error getting optimization data: {e}");
            report {"nodes": [], "error": str(e)};
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
        has history: list = [];

        useEffect(
            lambda -> None { document.title = "CGinS";link = document.querySelector(
                "link[rel*='icon']"
            )
            or document.createElement("link");link.type = "image/svg+xml";link.rel = "shortcut icon";link.href = "data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>👻</text></svg>";document.head.appendChild(
                link
            );},
            []
        );
        # Client-side wrappers for walkers
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

        async def client_get_optimization_data(projectName: str, opName: str) -> dict {
            console.log(f"Spawning GetOptimizationData for {projectName}, {opName}");
            resp = await root spawn GetOptimizationData(projectName=projectName, opName=opName);
            console.log("GetOptimizationData response:", resp);
            return resp.reports[0] if resp.reports else {"nodes": []};
        }

        def navigate(path: str) -> None {
            console.log(f"Navigating to {path}");
            if page != path {
                history.push(page);
            }
            page = path;
            # window.history.pushState({}, "", path);
        }

        def go_back() -> None {
            if history.length > 0 {
                prev_page = history.pop();
                console.log(f"Going back to {prev_page}");
                page = prev_page;
            } else {
                navigate("/");
            }
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
                    else <Settings onNavigate={navigate} onReturn={go_back} />
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
                        getOpDataWalker={client_get_optimization_data}
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
