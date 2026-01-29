"""Main entry point for frontend."""

import from json { load, dump }
import from pathlib { Path }
import from base64 { b64decode }
import from os { path, makedirs, listdir }
import from datetime { datetime }
import from zipfile { ZipFile }
import sys;
import os;
import subprocess;

glob config_store: dict = {
         "llm_info": {"model": "", "apikey": "", "provider": "anthropic"}
     },
     config_path: str = Path("config.json"),
     repo_root: str = path.abspath("..");

def _project_dir(name: str) -> str {
    return path.join(repo_root, "projects", name);
}

def _ensure_dir(dir_path: str) -> None {
    if not path.exists(dir_path) {
        makedirs(dir_path);
    }
}

def _read_state(project_name: str) -> dict {
    state_path = path.join(_project_dir(project_name), "state.json");
    if not path.exists(state_path) {
        return {};
    }
    try {
        with open(state_path, "r") as f {
            return load(f);
        }
    } except Exception as e {
        print(f"Error reading state.json: {e}");
        return {};
    }
}

def _start_job(project_name: str, job_key: str, cmd: list, log_name: str) -> dict {
    project_dir = _project_dir(project_name);
    logs_dir = path.join(project_dir, "logs");
    _ensure_dir(logs_dir);
    state_path = path.join(project_dir, "state.json");
    log_path = path.join(logs_dir, log_name);

    runner = path.join(repo_root, "scripts", "run_job.py");
    full_cmd = [
        sys.executable,
        runner,
        "--state-path",
        state_path,
        "--job-key",
        job_key,
        "--log-path",
        log_path,
        "--cwd",
        repo_root,
        "--"
    ] + cmd;

    proc = subprocess.Popen(full_cmd, cwd=repo_root);
    return {"success": True, "pid": proc.pid, "log": log_path};
}

def _find_optimized_dir(project_name: str) -> str {
    base_dir = path.join(repo_root, "kernels", "optimized");
    if not path.exists(base_dir) {
        return "";
    }
    suffix = "_" + project_name;
    best_dir = "";
    best_time = 0.0;
    for name in listdir(base_dir) {
        full_path = path.join(base_dir, name);
        if not path.isdir(full_path) {
            continue;
        }
        if not name.endswith(suffix) {
            continue;
        }
        try {
            mtime = path.getmtime(full_path);
            if mtime > best_time {
                best_time = mtime;
                best_dir = full_path;
            }
        } except Exception:
            continue;
    }
    return best_dir;
}

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
    has projectName: str,
        code: str,
        weightsFileB64: str = "",
        validationSetPath: str = "";

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
                "validation_set": self.validationSetPath,
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

walker:pub GetProjectOps {
    has projectName: str;

    can return_ops with `root entry {
        try {
            base_dir = path.join("projects", self.projectName, "io", "individual_ops");
            if not path.exists(base_dir) {
                report {"ops": []};
                disengage;
            }

            ops = [];
            for name in listdir(base_dir) {
                full_path = path.join(base_dir, name);
                if path.isdir(full_path) {
                    count = len(
                        [
                            f
                            for f in listdir(full_path)
                            if f.startswith("entry_") and f.endswith(".pt")
                        ]
                    );
                    ops.append({"name": name, "count": count});
                }
            }

            report {"ops": ops};
        } except Exception as e {
            print(f"Error listing ops: {e}");
            report {"ops": []};
        }
    }
}

walker:pub GetProjectStatus {
    has projectName: str;

    can return_status with `root entry {
        state = _read_state(self.projectName);
        report state;
    }
}

walker:pub StartProfile {
    has projectName: str;

    can run with `root entry {
        cmd = [
            sys.executable,
            path.join(repo_root, "benchmarks", "profiler", "profile_project.py"),
            "--project",
            self.projectName
        ];
        result = _start_job(self.projectName, "profile", cmd, "profile.log");
        report result;
    }
}

walker:pub StartGenerate {
    has projectName: str;

    can run with `root entry {
        io_dir = path.join(_project_dir(self.projectName), "io", "individual_ops");
        out_dir = path.join(_project_dir(self.projectName), "kernels", "generated");
        _ensure_dir(out_dir);
        cmd = [
            sys.executable,
            "-m",
            "src.generator.main",
            "--io-dir",
            io_dir,
            "--out-dir",
            out_dir
        ];
        result = _start_job(self.projectName, "generate", cmd, "generate.log");
        report result;
    }
}

walker:pub StartOptimize {
    has projectName: str;

    can run with `root entry {
        io_dir = path.join(_project_dir(self.projectName), "io", "individual_ops");
        kernel_dir = path.join(_project_dir(self.projectName), "kernels", "generated", "individual_op_kernels");
        if not path.exists(kernel_dir) {
            kernel_dir = path.join(repo_root, "kernels", "generated", "individual_op_kernels");
        }
        cmd = [
            sys.executable,
            "-m",
            "src.optimizer.optimize_ops",
            io_dir,
            self.projectName,
            "--kernel-dir",
            kernel_dir
        ];
        result = _start_job(self.projectName, "optimize", cmd, "optimize.log");
        report result;
    }
}

walker:pub StartBenchmark {
    has projectName: str;

    can run with `root entry {
        cmd = [
            sys.executable,
            path.join(repo_root, "scripts", "benchmark_project.py"),
            "--project",
            self.projectName
        ];
        result = _start_job(self.projectName, "benchmark", cmd, "benchmark.log");
        report result;
    }
}

walker:pub ExportProject {
    has projectName: str;

    can run with `root entry {
        optimized_dir = _find_optimized_dir(self.projectName);
        source_dir = optimized_dir;
        if not source_dir {
            project_generated = path.join(_project_dir(self.projectName), "kernels", "generated", "individual_op_kernels");
            if path.exists(project_generated) {
                source_dir = project_generated;
            } else {
                source_dir = path.join(repo_root, "kernels", "generated", "individual_op_kernels");
            }
        }
        if not path.exists(source_dir) {
            report {"success": False, "error": "No kernels found to export."};
            disengage;
        }

        export_dir = path.join(_project_dir(self.projectName), "exports");
        _ensure_dir(export_dir);
        export_path = path.join(export_dir, self.projectName + ".cgins.zip");

        manifest = {
            "project": self.projectName,
            "created_at": datetime.now().isoformat(),
            "source_dir": source_dir,
            "kernels": []
        };

        with ZipFile(export_path, "w") as zipf {
            for root_dir, dirs, files in os.walk(source_dir) {
                for fname in files {
                    if fname.endswith(".cu") or fname.endswith(".so") or fname.endswith(".txt") {
                        full_path = path.join(root_dir, fname);
                        rel_path = path.relpath(full_path, source_dir);
                        zipf.write(full_path, path.join("kernels", rel_path));
                    }
                }
            }

            manifest_path = path.join(export_dir, "manifest.json");
            with open(manifest_path, "w") as f {
                dump(manifest, f, indent=2);
            }
            zipf.write(manifest_path, "manifest.json");
        }

        report {"success": True, "path": export_path};
    }
}

cl {
    import from react { useState, useEffect }
    import from sonner { Toaster }
    import from .contexts.ConfigContext { ConfigProvider }
    import from .components.Home { Home }
    import from .components.Project { Project }
    import from .components.Settings { Settings }
    import from .components.NewProject { NewProject }
    import from .components.OpDetails { OpDetails }

    import "./styles.css";

    def:pub app -> any {
        has page: str = "/";

        useEffect(lambda -> None { document.title = "CGinS";}, []);

        def base_route -> str {
            path_name = window.location.pathname;
            if path_name.startsWith("/app") {
                return "/app";
            }
            return "";
        }

        def normalize_path(raw: str) -> str {
            base = base_route();
            if base and raw.startsWith(base) {
                local_path = raw.slice(base.length);
                if local_path == "" {
                    local_path = "/";
                }
                return local_path;
            }
            return raw or "/";
        }

        def to_full_path(raw: str) -> str {
            base = base_route();
            if base {
                if raw == "/" or raw == "" {
                    return base;
                }
                return base + raw;
            }
            return raw;
        }

        useEffect(
            lambda -> None { def sync_from_location -> None {
                page = normalize_path(window.location.pathname);
            } sync_from_location();window.addEventListener(
                "popstate", sync_from_location
            );return lambda -> None { window.removeEventListener(
                "popstate", sync_from_location
            );}; },
            []
        );

        # Client-side wrappers for walkers
        async def client_get_config -> dict {
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

        async def client_get_projects -> dict {
            resp = await root spawn GetProjects();
            return resp.reports[0] if resp.reports else {"projects": []};
        }

        async def client_get_project_ops(projectName: str) -> dict {
            resp = await root spawn GetProjectOps(projectName=projectName);
            return resp.reports[0] if resp.reports else {"ops": []};
        }

        async def client_get_project_status(projectName: str) -> dict {
            resp = await root spawn GetProjectStatus(projectName=projectName);
            return resp.reports[0] if resp.reports else {};
        }

        async def client_start_profile(projectName: str) -> dict {
            resp = await root spawn StartProfile(projectName=projectName);
            return resp.reports[0] if resp.reports else {};
        }

        async def client_start_generate(projectName: str) -> dict {
            resp = await root spawn StartGenerate(projectName=projectName);
            return resp.reports[0] if resp.reports else {};
        }

        async def client_start_optimize(projectName: str) -> dict {
            resp = await root spawn StartOptimize(projectName=projectName);
            return resp.reports[0] if resp.reports else {};
        }

        async def client_start_benchmark(projectName: str) -> dict {
            resp = await root spawn StartBenchmark(projectName=projectName);
            return resp.reports[0] if resp.reports else {};
        }

        async def client_export_project(projectName: str) -> dict {
            resp = await root spawn ExportProject(projectName=projectName);
            return resp.reports[0] if resp.reports else {};
        }

        def navigate(path: str) -> None {
            target = normalize_path(path);
            full_path = to_full_path(target);
            window.history.pushState({}, "", full_path);
            page = target;
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
                    else <Project
                        name={project_name}
                        onNavigate={navigate}
                        getProjectOpsWalker={client_get_project_ops}
                        getProjectStatusWalker={client_get_project_status}
                        startProfileWalker={client_start_profile}
                        startGenerateWalker={client_start_generate}
                        startOptimizeWalker={client_start_optimize}
                        startBenchmarkWalker={client_start_benchmark}
                        exportProjectWalker={client_export_project}
                    />
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
