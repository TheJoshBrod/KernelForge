"""Main entry point for frontend."""

import from json { load, dump }
import from pathlib { Path }
import from base64 { b64decode, b64encode }
import from os { path, makedirs, listdir }
import from datetime { datetime }
import from zipfile { ZipFile }
import sys;
import os;
import subprocess;

glob config_store: dict = {
         "llm_info": {"model": "", "apikey": "", "provider": ""},
         "providers": []
     },
     config_path: str = "config.json",
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

def _update_job_state(project_name: str, job_key: str, updates: dict) -> None {
    state_path = path.join(_project_dir(project_name), "state.json");
    state = _read_state(project_name);
    job_state = state[job_key] if job_key in state else {};
    for k in updates {
        job_state[k] = updates[k];
    }
    state[job_key] = job_state;
    with open(state_path, "w") as f {
        dump(state, f, indent=2);
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
        if path.isdir(full_path) and name.endswith(suffix) {
            try {
                mtime = path.getmtime(full_path);
                if mtime > best_time {
                    best_time = mtime;
                    best_dir = full_path;
                }
            } except Exception as e {
                print(f"Error reading optimized dir: {e}");
            }
        }
    }
    return best_dir;
}

def _export_project_zip(project_name: str, ops: list = None) -> dict {
    optimized_dir = _find_optimized_dir(project_name);
    source_dir = optimized_dir;
    if not source_dir {
        project_generated = path.join(
            _project_dir(project_name), "kernels", "generated", "individual_op_kernels"
        );
        if path.exists(project_generated) {
            source_dir = project_generated;
        } else {
            source_dir = path.join(
                repo_root, "kernels", "generated", "individual_op_kernels"
            );
        }
    }

    if not path.exists(source_dir) {
        return {"success": False, "error": "No kernels found to export."};
    }

    selected_ops = ops or [];
    if not selected_ops {
        selected_ops = [];
        for name in listdir(source_dir) {
            full_path = path.join(source_dir, name);
            if path.isdir(full_path) {
                selected_ops.append(name);
            }
        }
    }

    export_dir = path.join(_project_dir(project_name), "exports");
    _ensure_dir(export_dir);
    export_path = path.join(export_dir, project_name + ".cgins.zip");

    manifest = {
        "project": project_name,
        "created_at": datetime.now().isoformat(),
        "source_dir": source_dir,
        "selected_ops": selected_ops,
        "kernels": []
    };

    with ZipFile(export_path, "w") as zipf {
        for op_name in selected_ops {
            op_dir = path.join(source_dir, op_name);
            if not path.exists(op_dir) {
                continue;
            }
            for entry in os.walk(op_dir) {
                root_dir = entry[0];
                files = entry[2];
                for fname in files {
                    if fname.endswith(".cu")
                    or fname.endswith(".so")
                    or fname.endswith(".txt")
                    or fname.endswith(".log")
                    or fname.endswith(".json") {
                        full_path = path.join(root_dir, fname);
                        rel_path = path.relpath(full_path, source_dir);
                        zipf.write(full_path, path.join("kernels", rel_path));
                    }
                }
            }
        }
        manifest_path = path.join(export_dir, "manifest.json");
        with open(manifest_path, "w") as f {
            dump(manifest, f, indent=2);
        }
        zipf.write(manifest_path, "manifest.json");
    }

    return {"success": True, "path": export_path};
}

def _file_kind(fname: str) -> str {
    if fname.endswith(".cu")
    or fname.endswith(".txt")
    or fname.endswith(".log")
    or fname.endswith(".json") {
        return "text";
    }
    return "binary";
}

def _file_info(full_path: str) -> dict {
    return {
        "name": path.basename(full_path),
        "path": path.relpath(full_path, repo_root),
        "size": path.getsize(full_path),
        "mtime": path.getmtime(full_path),
        "kind": _file_kind(path.basename(full_path))
    };
}

def _list_files(dir_path: str, exts: list = None, max_files: int = 200) -> list {
    files = [];
    if not path.exists(dir_path) {
        return files;
    }
    count = 0;
    for fname in listdir(dir_path) {
        full_path = path.join(dir_path, fname);
        if path.isdir(full_path) {
            continue;
        }
        if exts {
            ok = False;
            for ext in exts {
                if fname.endswith(ext) {
                    ok = True;
                }
            }
            if not ok {
                continue;
            }
        }
        files.append(_file_info(full_path));
        count = count + 1;
        if count >= max_files {
            break;
        }
    }
    return files;
}

def _is_allowed_repo_path(rel_path: str, project_name: str) -> bool {
    if path.isabs(rel_path) {
        return False;
    }
    clean = path.normpath(rel_path);
    if clean.startswith("..") {
        return False;
    }
    allowed = [
        path.join("projects", project_name),
        path.join("kernels", "optimized"),
        path.join("kernels", "generated")
    ];
    for prefix in allowed {
        if clean.startswith(prefix) {
            return True;
        }
    }
    return False;
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
        validationSetPath: str = "",
        validationZipB64: str = "",
        validationZipName: str = "";

    can create with `root entry {
        try {
            # Define projects directory
            projects_dir = path.join(repo_root, "projects");
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

            validation_dir = self.validationSetPath;
            if self.validationZipB64 {
                data_dir = path.join(project_path, "data");
                _ensure_dir(data_dir);
                zip_path = path.join(data_dir, "validation.zip");
                try {
                    zip_bytes = b64decode(self.validationZipB64);
                    with open(zip_path, "wb") as f {
                        f.write(zip_bytes);
                    }
                    validation_extract = path.join(data_dir, "validation");
                    _ensure_dir(validation_extract);
                    with ZipFile(zip_path, "r") as zipf {
                        zipf.extractall(validation_extract);
                    }
                    validation_dir = path.relpath(validation_extract, project_path);
                } except Exception as e {
                    print(f"Error extracting validation zip: {e}");
                }
            }

            # Save config.json
            config = {
                "validation_dir": validation_dir,
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
            projects_dir = path.join(repo_root, "projects");
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
            base_dir = path.join(
                repo_root, "projects", self.projectName, "io", "individual_ops"
            );
            if not path.exists(base_dir) {
                report {"ops": []};
                disengage;
            }

            ops = [];
            for name in listdir(base_dir) {
                full_path = path.join(base_dir, name);
                if path.isdir(full_path) {
                    count = 0;
                    for f in listdir(full_path) {
                        if f.startswith("entry_") and f.endswith(".pt") {
                            count = count + 1;
                        }
                    }
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

walker:pub SetJobControl {
    has projectName: str,
        jobKey: str,
        action: str;

    can run with `root entry {
        try {
            action = self.action.lower();
            updates = {};
            if action == "pause" {
                updates = {"control": "paused", "status": "paused"};
            } elif action == "resume" {
                updates = {"control": "running", "status": "running"};
            } elif action == "cancel" {
                updates = {"control": "cancelled", "status": "cancelled"};
            } else {
                report {"success": False, "error": "Unknown action"};
                disengage;
            }
            _update_job_state(self.projectName, self.jobKey, updates);
            report {"success": True};
        } except Exception as e {
            print(f"Error setting job control: {e}");
            report {"success": False, "error": "Failed to update job state"};
        }
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
        kernel_dir = path.join(
            _project_dir(self.projectName),
            "kernels",
            "generated",
            "individual_op_kernels"
        );
        if not path.exists(kernel_dir) {
            kernel_dir = path.join(
                repo_root, "kernels", "generated", "individual_op_kernels"
            );
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
            path.join(repo_root, "scripts", "benchmark_project_ops.py"),
            "--project",
            self.projectName
        ];
        result = _start_job(self.projectName, "benchmark", cmd, "benchmark.log");
        report result;
    }
}

walker:pub ExportProject {
    has projectName: str,
        ops: list = [];

    can run with `root entry {
        result = _export_project_zip(self.projectName, self.ops);
        report result;
    }
}

walker:pub GetOpDetails {
    has projectName: str,
        opName: str;

    can run with `root entry {
        project_dir = _project_dir(self.projectName);
        profile_dir = path.join(project_dir, "io", "individual_ops", self.opName);
        entry_files = [];
        entry_count = 0;
        if path.exists(profile_dir) {
            for fname in listdir(profile_dir) {
                if fname.startswith("entry_") and fname.endswith(".pt") {
                    entry_count = entry_count + 1;
                    if entry_count <= 50 {
                        entry_files.append(_file_info(path.join(profile_dir, fname)));
                    }
                }
            }
        }

        generated_dir = path.join(
            project_dir, "kernels", "generated", "individual_op_kernels", self.opName
        );
        gen_kernel_dir = path.join(generated_dir, "kernel");
        gen_attempt_dir = path.join(generated_dir, "attempts");
        gen_kernels = _list_files(gen_kernel_dir, [".cu", ".so"]);
        gen_logs = _list_files(gen_attempt_dir, [".txt", ".log"]);
        gen_best_path = path.join(generated_dir, "kernel.cu");
        gen_best = _file_info(gen_best_path) if path.exists(gen_best_path) else {};

        opt_base = _find_optimized_dir(self.projectName);
        if not opt_base {
            opt_base = path.join(repo_root, "kernels", "optimized");
        }
        opt_dir = path.join(opt_base, self.opName) if opt_base else "";
        opt_kernels = _list_files(opt_dir, [".cu", ".so"]);
        opt_logs = _list_files(opt_dir, [".txt", ".log"]);

        report {
            "op": self.opName,
            "profile": {
                "count": entry_count,
                "entries": entry_files,
                "dir": path.relpath(profile_dir, repo_root)
                if path.exists(profile_dir)
                else ""
            },
            "generated": {
                "dir": path.relpath(generated_dir, repo_root)
                if path.exists(generated_dir)
                else "",
                "kernels": gen_kernels,
                "logs": gen_logs,
                "best": gen_best
            },
            "optimized": {
                "dir": path.relpath(opt_dir, repo_root)
                if opt_dir and path.exists(opt_dir)
                else "",
                "kernels": opt_kernels,
                "logs": opt_logs
            }
        };
    }
}

walker:pub ReadRepoFile {
    has projectName: str,
        relPath: str;

    can run with `root entry {
        try {
            if not _is_allowed_repo_path(self.relPath, self.projectName) {
                report {"success": False, "error": "Path not allowed"};
                disengage;
            }
            safe_rel = path.normpath(self.relPath);
            full_path = path.join(repo_root, safe_rel);
            if not path.exists(full_path) {
                report {"success": False, "error": "File not found"};
                disengage;
            }
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f {
                content = f.read();
            }
            report {
                "success": True,
                "content": content,
                "name": path.basename(full_path)
            };
        } except Exception as e {
            print(f"Error reading file: {e}");
            report {"success": False, "error": "Failed to read file"};
        }
    }
}

walker:pub DownloadExport {
    has projectName: str,
        ops: list = [];

    can run with `root entry {
        if self.ops and self.ops.length > 0 {
            export_result = _export_project_zip(self.projectName, self.ops);
            if not export_result["success"] {
                report export_result;
                disengage;
            }
            export_path = export_result["path"];
        } else {
            export_path = path.join(
                _project_dir(self.projectName),
                "exports",
                self.projectName + ".cgins.zip"
            );
            if not path.exists(export_path) {
                export_result = _export_project_zip(self.projectName, []);
                if not export_result["success"] {
                    report export_result;
                    disengage;
                }
                export_path = export_result["path"];
            }
        }
        with open(export_path, "rb") as f {
            data = f.read();
        }
        b64 = b64encode(data).decode("utf-8");
        report {"success": True, "name": path.basename(export_path), "data": b64};
    }
}

walker:pub GetProjectBenchmarks {
    has projectName: str;

    can run with `root entry {
        bench_path = path.join(
            _project_dir(self.projectName), "benchmarks", "op_benchmarks.json"
        );
        if not path.exists(bench_path) {
            report {"results": [], "winners": []};
            disengage;
        }
        try {
            with open(bench_path, "r") as f {
                data = load(f);
            }
            results = data["results"] if data and "results" in data else [];
            winners = [];
            for r in results {
                if r and r["winner"] {
                    winners.append(r["op"]);
                }
            }
            report {"results": results, "winners": winners};
        } except Exception as e {
            print(f"Error reading benchmarks: {e}");
            report {"results": [], "winners": []};
        }
    }
}

cl import from react { useEffect }
cl import from sonner { Toaster }
cl import from "react-router-dom" { BrowserRouter as Router, Routes, Route }
cl import from .ConfigContext { ConfigProvider }
cl import from .Home { Home }
cl import from .Project { Project }
cl import from .Settings { Settings }
cl import from .NewProject { NewProject }
cl import from .OpDetails { OpDetails }
cl import "./styles.css";

cl {
    def NotFound -> any {
        return
            <div className="flex min-h-screen flex-col items-center justify-center">
                <h1>
                    404 Not Found
                </h1>
                <p className="text-zinc-500 mt-2">
                    The page you are looking for does not exist.
                </p>
            </div>;
    }

    def:pub app -> any {
        useEffect(lambda -> None { document.title = "CGinS";}, []);
        return
            <Router>
                <ConfigProvider>
                    <div
                        className="bg-zinc-950 min-h-screen text-zinc-100 font-sans antialiased"
                    >
                        <Routes>
                            <Route path="/" element={<Home />} />
                            <Route path="/settings" element={<Settings />} />
                            <Route path="/new-project" element={<NewProject />} />
                            <Route
                                path="/project/:projectName"
                                element={<Project />}
                            />
                            <Route
                                path="/project/:projectName/:opName"
                                element={<OpDetails />}
                            />
                            <Route path="*" element={<NotFound />} />
                        </Routes>
                        <Toaster richColors={True} theme="dark" />
                    </div>
                </ConfigProvider>
            </Router>;
    }
}
