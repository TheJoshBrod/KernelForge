"""Main entry point for frontend."""
import from json { load, loads, dump }
import from pathlib { Path }
import from base64 { b64decode, b64encode }
import from os { path, makedirs, listdir }
import from datetime { datetime }
import socket;
import paramiko;
import torch;

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

def _apply_llm_env_from_config(cfg: dict) -> None {
    if not cfg {
        return;
    }
    llm = cfg["llm_info"] if cfg and "llm_info" in cfg else {};
    provider = llm["provider"] if llm and "provider" in llm else "";
    model = llm["model"] if llm and "model" in llm else "";
    apikey = llm["apikey"] if llm and "apikey" in llm else "";

    if provider {
        os.environ["LLM_PROVIDER"] = provider;
    }
    if provider == "openai" {
        if apikey {
            os.environ["OPENAI_API_KEY"] = apikey;
        }
        if model {
            os.environ["OPENAI_MODEL"] = model;
        }
    } elif provider == "anthropic" {
        if apikey {
            os.environ["ANTHROPIC_API_KEY"] = apikey;
        }
        if model {
            os.environ["ANTHROPIC_MODEL"] = model;
        }
    } elif provider == "gemini" {
        if apikey {
            os.environ["GOOGLE_API_KEY"] = apikey;
            os.environ["GEMINI_API_KEY"] = apikey;
        }
        if model {
            os.environ["GEMINI_MODEL"] = model;
        }
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

def _start_job_container(
    project_name: str, job_key: str, cmd: list, log_name: str, image: str
) -> dict {
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
        "--use-container",
        "--container-image",
        image,
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

def _detect_hardware -> dict {
    try {
        code = "import json, torch; info={'cuda_available': torch.cuda.is_available(), 'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0, 'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else '', 'cuda_version': torch.version.cuda or '', 'mps_available': getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available(), 'mps_built': getattr(torch.backends, 'mps', None) and torch.backends.mps.is_built()}; print(json.dumps(info))";
        out = subprocess.check_output([sys.executable, "-c", code], cwd=repo_root);
        text = out.decode("utf-8").strip();
        if text {
            return loads(text);
        }
    } except Exception as e {
        return {"cuda_available": False, "mps_available": False, "error": str(e)};
    }
    return {"cuda_available": False, "mps_available": False};
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

def _run_llm_test(provider: str, model: str, apikey: str) -> dict {
    if not provider {
        return {"success": False, "error": "Missing provider"};
    }
    if not model {
        return {"success": False, "error": "Missing model"};
    }
    if not apikey {
        return {"success": False, "error": "Missing API key"};
    }
    test_script = path.join(repo_root, "scripts", "test_llm_connection.py");
    cmd = [
        sys.executable,
        test_script,
        "--provider",
        provider,
        "--model",
        model,
        "--apikey",
        apikey
    ];
    try {
        result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True);
        stdout = result.stdout.strip();
        stderr = result.stderr.strip();
        if stdout {
            try {
                data = loads(stdout);
            } except Exception as e {
                data = {
                    "success": False,
                    "error": "Failed to parse response",
                    "raw": stdout,
                    "exception": str(e)
                };
            }
        } else {
            data = {"success": False, "error": "No output from test"};
        }
        if stderr and "stderr" not in data {
            data["stderr"] = stderr;
        }
        data["return_code"] = result.returncode;
        return data;
    } except Exception as e {
        return {"success": False, "error": str(e)};
    }
}

glob config_path: str = "config.json";

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

walker:pub TestLLMConnection {
    has cfg: dict = {};

    can run with `root entry {
        cfg = self.cfg;
        if not cfg {
            try {
                with open(config_path, "r") as f {
                    cfg = load(f);
                }
            } except Exception as e {
                print(f"Error loading config: {e}");
                cfg = config_store;
            }
        }
        llm = cfg["llm_info"]
        if cfg and "llm_info" in cfg
        else config_store["llm_info"];
        provider = llm["provider"] if llm and "provider" in llm else "";
        model = llm["model"] if llm and "model" in llm else "";
        apikey = llm["apikey"] if llm and "apikey" in llm else "";
        result = _run_llm_test(provider, model, apikey);
        report result;
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
            # Save model.py
            if self.code {
                with open(path.join(project_path, "model.py"), "w") as f {
                    f.write(self.code);
                }
            }

            # Save weights.pt
            # Save weights.pt
            if self.weightsFileB64 {
                weights_data = b64decode(self.weightsFileB64);
                with open(path.join(project_path, "weights.pt"), "wb") as f {
                    f.write(weights_data);
                }
            }

            validation_dir = self.validationSetPath;
            if self.validationZipB64 {
                validation_dir = path.join("data", "validation");
            }

            # Save config.json
            # Save config.json
            hw = _detect_hardware();
            target_device = "cpu";
            if hw and "mps_available" in hw and hw["mps_available"] {
                target_device = "mps";
            } elif hw and "cuda_available" in hw and hw["cuda_available"] {
                target_device = "cuda";
            }

            config = {
                "validation_dir": validation_dir,
                "created_at": datetime.now().isoformat(),
                "generator": {
                    "skip_ops": [],
                    "only_ops": [],
                    "max_ops": 0,
                    "use_baseline_kernels": True,
                    "use_baseline_as_template": True,
                    "max_attempts": 8,
                    "extra_validation_cases": 1,
                    "target_device": target_device
                },
                "profile": {"allow_ops": [], "skip_ops": [], "skip_prefixes": []}
            };
            with open(path.join(project_path, "config.json"), "w") as f {
                dump(config, f, indent=2);
            }

            prepare_result = None;
            profile_result = None;
            has_assets = False;
            uploads_dir = path.join(project_path, ".uploads");
            _ensure_dir(uploads_dir);

            weights_b64_path = "";
            validation_b64_path = "";
            validation_name_path = "";
            if self.weightsFileB64 {
                weights_b64_path = path.join(uploads_dir, "weights.b64");
                with open(weights_b64_path, "w") as f {
                    f.write(self.weightsFileB64);
                }
                has_assets = True;
            }
            if self.validationZipB64 {
                validation_b64_path = path.join(uploads_dir, "validation.b64");
                with open(validation_b64_path, "w") as f {
                    f.write(self.validationZipB64);
                }
                validation_name_path = path.join(uploads_dir, "validation.name");
                with open(validation_name_path, "w") as f {
                    f.write(self.validationZipName);
                }
                has_assets = True;
            }

            if has_assets {
                _update_job_state(
                    self.projectName,
                    "profile",
                    {"status": "queued", "message": "Preparing assets"}
                );
                prepare_cmd = [
                    sys.executable,
                    path.join(repo_root, "scripts", "prepare_project_assets.py"),
                    "--project",
                    self.projectName,
                    "--weights-b64-path",
                    weights_b64_path,
                    "--validation-b64-path",
                    validation_b64_path,
                    "--validation-name-path",
                    validation_name_path
                ];
                prepare_result = _start_job(
                    self.projectName, "prepare", prepare_cmd, "prepare.log"
                );
            } else {
                profile_cmd = [
                    sys.executable,
                    path.join(
                        repo_root, "benchmarks", "profiler", "profile_project.py"
                    ),
                    "--project",
                    self.projectName
                ];
                profile_result = _start_job(
                    self.projectName, "profile", profile_cmd, "profile.log"
                );
            }

            report {
                "success": True,
                "name": self.projectName,
                "profile": profile_result,
                "prepare": prepare_result
            };
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

walker:pub GetSystemInfo {
    has host: str = "";
    has user: str = "";
    has password: str = "";
    has key_path: str = "";

    can return_info with `root entry {
        try {
            all_gpus = [];

            # 1. LOCAL CHECK
            # 1. LOCAL CHECK
            try {
                import torch;
                if torch.cuda.is_available() {
                    count = torch.cuda.device_count();
                    for i in range(count) {
                        device_name = torch.cuda.get_device_name(i);
                        props = torch.cuda.get_device_properties(i);
                        total_memory = props.total_memory / (1024 ** 3);

                        arch_name = "Unknown";
                        if props.major == 7 {
                            arch_name = "Volta/Turing";
                        } elif props.major == 8 {
                            arch_name = "Ampere";
                        } elif props.major == 9 {
                            arch_name = "Hopper";
                        }

                        all_gpus.append(
                            {
                                "name": f"{device_name} (Local)",
                                "memory": f"{total_memory:.1f} GB",
                                "compute_capability": f"v{props.major}.{props.minor} {arch_name}",
                                "sm_count": props.multi_processor_count
                            }
                        );
                    }
                }
            } except Exception as e {
                print(f"Local GPU check failed: {e}");
            }

            # 2. REMOTE CHECK VIA SSH
            # 2. REMOTE CHECK VIA SSH
            # 2. REMOTE CHECK VIA SSH
            # 2. REMOTE CHECK VIA SSH
            # 2. REMOTE CHECK VIA SSH
            # 2. REMOTE CHECK VIA SSH
            # 2. REMOTE CHECK VIA SSH
            # 2. REMOTE CHECK VIA SSH
            # 2. REMOTE CHECK VIA SSH
            # 2. REMOTE CHECK VIA SSH
            # 2. REMOTE CHECK VIA SSH
            # 2. REMOTE CHECK VIA SSH
            # 2. REMOTE CHECK VIA SSH
            # 2. REMOTE CHECK VIA SSH
            # 2. REMOTE CHECK VIA SSH
            # 2. REMOTE CHECK VIA SSH
            if self.host and self.user {
                try {
                    client = paramiko.SSHClient();
                    client.set_missing_host_key_policy(paramiko.AutoAddPolicy());

                    # Resolve key path
                    # Resolve key path
                    # Resolve key path
                    # Resolve key path
                    # Resolve key path
                    # Resolve key path
                    # Resolve key path
                    # Resolve key path
                    # Resolve key path
                    # Resolve key path
                    # Resolve key path
                    # Resolve key path
                    # Resolve key path
                    # Resolve key path
                    # Resolve key path
                    # Resolve key path
                    k_path = None;
                    if self.key_path {
                        if path.isdir(self.key_path) {
                            possible = ["id_rsa", "id_ed25519"];
                            for p in possible {
                                fp = path.join(self.key_path, p);
                                if path.exists(fp) {
                                    k_path = fp;
                                    break;
                                }
                            }
                        } else {
                            k_path = self.key_path;
                        }
                    }

                    client.connect(
                        hostname=self.host,
                        username=self.user,
                        password=self.password,
                        key_filename=k_path,
                        timeout=5
                    );

                    # Python script to check GPU
                    # Python script to check GPU
                    # Python script to check GPU
                    # Python script to check GPU
                    # Python script to check GPU
                    # Python script to check GPU
                    # Python script to check GPU
                    # Python script to check GPU
                    # Python script to check GPU
                    # Python script to check GPU
                    # Python script to check GPU
                    # Python script to check GPU
                    # Python script to check GPU
                    # Python script to check GPU
                    # Python script to check GPU
                    # Python script to check GPU
                    cmd_script = """
import json
import sys
import subprocess

def get_gpu_info():
    # Try torch first
    try:
        import torch
        if torch.cuda.is_available():
            gpus = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / (1024**3)

                arch_name = "Unknown"
                if props.major == 7: arch_name = "Volta/Turing"
                elif props.major == 8: arch_name = "Ampere"
                elif props.major == 9: arch_name = "Hopper"

                gpus.append({
                    'name': torch.cuda.get_device_name(i),
                    'memory': f"{mem_gb:.1f} GB",
                    'compute_capability': f"v{props.major}.{props.minor} {arch_name}",
                    'sm_count': props.multi_processor_count
                })
            return gpus
    except ImportError as e:
        sys.stderr.write(f"DEBUG: Torch import failed: {e}\\n")
    except Exception as e:
        sys.stderr.write(f"DEBUG: Torch check failed: {e}\\n")

    # Fallback to nvidia-smi
    try:
        sys.stderr.write("DEBUG: Falling back to nvidia-smi\\n")
        cmd = ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"]
        output = subprocess.check_output(cmd).decode('utf-8').strip()
        gpus = []
        for line in output.split('\\n'):
            if not line.strip(): continue
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 3:
                name = parts[0]
                mem_str = parts[1] # e.g. "24576 MiB"
                cc = parts[2] # e.g. "8.6"

                # Convert memory to GB
                try:
                    mem_val = float(mem_str.split()[0])
                    mem_gb = f"{mem_val / 1024:.1f} GB"
                except:
                    mem_gb = mem_str

                # Arch name from CC
                arch_name = "Unknown"
                try:
                    major = int(str(cc).split('.')[0])
                    if major == 7: arch_name = "Volta/Turing"
                    elif major == 8: arch_name = "Ampere"
                    elif major == 9: arch_name = "Hopper"
                except:
                    pass

                gpus.append({
                    "name": name,
                    "memory": mem_gb,
                    "compute_capability": f"v{cc} {arch_name}",
                    "sm_count": "N/A (No Torch)"
                })
        return gpus
    except Exception:
        return []

print(json.dumps(get_gpu_info()))
""";

                    b64_script = b64encode(cmd_script.encode('utf-8')).decode('utf-8');
                    # Use bash -lc to ensure profile/bashrc are sourced (PATH, virtualenvs, etc.)
                    # Use bash -lc to ensure profile/bashrc are sourced (PATH, virtualenvs, etc.)
                    # Use bash -lc to ensure profile/bashrc are sourced (PATH, virtualenvs, etc.)
                    # Use bash -lc to ensure profile/bashrc are sourced (PATH, virtualenvs, etc.)
                    # Use bash -lc to ensure profile/bashrc are sourced (PATH, virtualenvs, etc.)
                    # Use bash -lc to ensure profile/bashrc are sourced (PATH, virtualenvs, etc.)
                    # Use bash -lc to ensure profile/bashrc are sourced (PATH, virtualenvs, etc.)
                    # Use bash -lc to ensure profile/bashrc are sourced (PATH, virtualenvs, etc.)
                    # Use bash -lc to ensure profile/bashrc are sourced (PATH, virtualenvs, etc.)
                    # Use bash -lc to ensure profile/bashrc are sourced (PATH, virtualenvs, etc.)
                    # Use bash -lc to ensure profile/bashrc are sourced (PATH, virtualenvs, etc.)
                    # Use bash -lc to ensure profile/bashrc are sourced (PATH, virtualenvs, etc.)
                    # Use bash -lc to ensure profile/bashrc are sourced (PATH, virtualenvs, etc.)
                    # Use bash -lc to ensure profile/bashrc are sourced (PATH, virtualenvs, etc.)
                    # Use bash -lc to ensure profile/bashrc are sourced (PATH, virtualenvs, etc.)
                    # Use bash -lc to ensure profile/bashrc are sourced (PATH, virtualenvs, etc.)
                    # We escape single quotes for the bash command
                    # We escape single quotes for the bash command
                    # We escape single quotes for the bash command
                    # We escape single quotes for the bash command
                    # We escape single quotes for the bash command
                    # We escape single quotes for the bash command
                    # We escape single quotes for the bash command
                    # We escape single quotes for the bash command
                    # We escape single quotes for the bash command
                    # We escape single quotes for the bash command
                    # We escape single quotes for the bash command
                    # We escape single quotes for the bash command
                    # We escape single quotes for the bash command
                    # We escape single quotes for the bash command
                    # We escape single quotes for the bash command
                    # We escape single quotes for the bash command
                    cmd = f"""bash -lc 'python3 -c "import base64; exec(base64.b64decode(\\"{b64_script}\\"))"'""";

                    exec_res = client.exec_command(cmd);
                    stdout = exec_res[1];
                    stderr = exec_res[2];

                    exit_status = stdout.channel.recv_exit_status();

                    if exit_status == 0 {
                        out_json = stdout.read().decode().strip();
                        remote_gpus = loads(out_json);
                        for r_gpu in remote_gpus {
                            r_gpu["name"] = f"{r_gpu['name']} (Remote: {self.host})";
                            all_gpus.append(r_gpu);
                        }
                    } else {
                        err = stderr.read().decode().strip();
                        print(f"Remote python failed: {err}");
                        # Don't fail completely, just log error and continue with local
                        # Don't fail completely, just log error and continue with local
                        # Don't fail completely, just log error and continue with local
                        # Don't fail completely, just log error and continue with local
                        # Don't fail completely, just log error and continue with local
                        # Don't fail completely, just log error and continue with local
                        # Don't fail completely, just log error and continue with local
                        # Don't fail completely, just log error and continue with local
                        # Don't fail completely, just log error and continue with local
                        # Don't fail completely, just log error and continue with local
                        # Don't fail completely, just log error and continue with local
                        # Don't fail completely, just log error and continue with local
                        # Don't fail completely, just log error and continue with local
                        # Don't fail completely, just log error and continue with local
                        # Don't fail completely, just log error and continue with local
                        # Don't fail completely, just log error and continue with local
                    }
                    client.close();
                } except Exception as e {
                    print(f"SSH Check failed: {e}");
                }
            }

            report {
                "available": True if all_gpus else False,
                "gpus": all_gpus,
                "source": "mixed"
            };
        } except Exception as e {
            report {"available": False, "error": str(e)};
        }
    }
}

walker:pub CheckSshConnection {
    has host: str;
    has user: str;
    has password: str = "";
    has key_path: str = "";
    has port: int = 22;
    has timeout: int = 5;

    can check with `root entry {
        client = paramiko.SSHClient();
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy());

        try {
            print(
                f"Attempting SSH connection to {self.user}@{self.host}:{self.port}..."
            );

            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            # We'll explicitly handle the key_filename argument logic
            k_path = None;
            if self.key_path {
                if path.isdir(self.key_path) {
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    # If user provided a directory (e.g. ~/.ssh/), try to find common keys
                    possible_keys = ["id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"];
                    found_keys = [];
                    for name in possible_keys {
                        full = path.join(self.key_path, name);
                        if path.exists(full) {
                            found_keys.append(full);
                        }
                    }
                    if found_keys {
                        k_path = found_keys;
                        print(f"Found keys in directory: {found_keys}");
                    } else {
                        print(
                            f"Warning: {self.key_path} is a directory but no common keys found inside."
                        );
                    }
                } else {
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    # It's a file
                    k_path = self.key_path;
                }
            }

            client.connect(
                hostname=self.host,
                port=self.port,
                username=self.user,
                password=self.password,
                key_filename=k_path,
                timeout=self.timeout
            );

            print("SSH connection established. Executing test command...");

            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            # Execute a simple command to verify shell access
            exec_res = client.exec_command('echo "Connection Successful"');
            stdin = exec_res[0];
            stdout = exec_res[1];
            stderr = exec_res[2];
            exit_status = stdout.channel.recv_exit_status();

            if exit_status == 0 {
                output = stdout.read().decode().strip();
                print(f"Command execution successful: {output}");
                report {
                    "success": True,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                };
            } else {
                error_msg = stderr.read().decode().strip();
                print(
                    f"Command execution failed with status {exit_status}: {error_msg}"
                );
                report {"success": False, "error": f"Command failed: {error_msg}"};
            }
        } except Exception as e {
            print(f"SSH/SFTP check failed for {self.host}: {e}");
            report {"success": False, "error": str(e)};
        } finally {
            client.close();
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

walker:pub StartGenerateAdvanced {
    has projectName: str,
        ops: list = [],
        optimize: bool = False,
        benchmark: bool = True,
        iterations: int = 3,
        patience: int = 2,
        minImproveMs: float = 0.0,
        minSpeedupPct: float = 5.0,
        genAttempts: int = 4,
        image: str = "cgins-codex:latest",
        targetDevice: str = "cuda";

    can run with `root entry {
        try {
            cfg = {};
            try {
                with open(config_path, "r") as f {
                    cfg = load(f);
                }
            } except Exception as e {
                print(f"Error loading config: {e}");
                cfg = config_store;
            }
            _apply_llm_env_from_config(cfg);

            target_device = (self.targetDevice or "cuda").lower();
            if target_device == "gpu" or target_device == "cuda" {
                target_device = "cuda";
            } elif target_device == "mps" {
                target_device = "mps";
            } elif target_device != "cpu" {
                target_device = "cuda";
            }
            os.environ["CGINS_TARGET_DEVICE"] = target_device;
            try {
                project_cfg_path = path.join(
                    _project_dir(self.projectName), "config.json"
                );
                project_cfg = {};
                if path.exists(project_cfg_path) {
                    with open(project_cfg_path, "r") as f {
                        project_cfg = load(f);
                    }
                }
                gen_cfg = project_cfg["generator"]
                if project_cfg and "generator" in project_cfg
                else {};
                gen_cfg["target_device"] = target_device;
                project_cfg["generator"] = gen_cfg;
                with open(project_cfg_path, "w") as f {
                    dump(project_cfg, f, indent=2);
                }
            } except Exception as e {
                print(f"Failed to persist target device: {e}");
            }

            ops_list = [];
            ops_total = 0;
            for o in self.ops {
                if o {
                    ops_list.append(str(o));
                    ops_total = ops_total + 1;
                }
            }
            if ops_total == 0 {
                _update_job_state(
                    self.projectName,
                    "generate_adv",
                    {"status": "error", "message": "No ops selected or discovered."}
                );
                report {"success": False, "error": "No ops selected or discovered."};
                disengage;
            }
            io_root = path.join(_project_dir(self.projectName), "io", "individual_ops");
            missing_inputs = [];
            missing_total = 0;
            for op_name in ops_list {
                op_dir = path.join(io_root, str(op_name));
                has_entry = False;
                if path.exists(op_dir) {
                    for fname in listdir(op_dir) {
                        if fname.endswith(".pt") {
                            has_entry = True;
                            break;
                        }
                    }
                }
                if not has_entry {
                    missing_inputs.append(op_name);
                    missing_total = missing_total + 1;
                }
            }
            if missing_total > 0 {
                error_msg = (
                    f"Missing input entries for {missing_total} ops. " + "Run Profile first."
                );
                _update_job_state(
                    self.projectName,
                    "generate_adv",
                    {"status": "error", "message": error_msg}
                );
                report {"success": False, "error": error_msg};
                disengage;
            }

            now_ts = datetime.now().timestamp();
            _update_job_state(
                self.projectName,
                "generate_adv",
                {
                    "status": "running",
                    "started_at": now_ts,
                    "message": f"Queued advanced generation ({target_device})",
                    "progress": {
                        "current": 0,
                        "total": ops_total,
                        "percent": 0.0,
                        "updated_at": now_ts
                    }
                }
            );

            ops_rel = path.join("projects", self.projectName, "ops_selected.json");
            ops_path = path.join(repo_root, ops_rel);
            try {
                with open(ops_path, "w") as f {
                    dump(ops_list, f, indent=2);
                }
            } except Exception as e {
                print(f"Failed to write ops list: {e}");
            }

            cmd = [
                "python3",
                "scripts/codex_advanced_generate.py",
                "--project",
                self.projectName,
                "--ops-file",
                ops_rel
            ];
            if self.optimize {
                cmd.append("--optimize");
            }
            if self.benchmark {
                cmd.append("--benchmark");
            }
            if self.iterations > 0 {
                cmd.append("--opt-iterations");
                cmd.append(str(self.iterations));
            }
            if self.patience > 0 {
                cmd.append("--opt-patience");
                cmd.append(str(self.patience));
            }
            if self.genAttempts > 0 {
                cmd.append("--generate-attempts");
                cmd.append(str(self.genAttempts));
            }
            cmd.append("--opt-min-improve-ms");
            cmd.append(str(self.minImproveMs));
            cmd.append("--opt-min-speedup-pct");
            cmd.append(str(self.minSpeedupPct));
            result = _start_job_container(
                self.projectName, "generate_adv", cmd, "generate_adv.log", self.image
            );
            report result;
        } except Exception as e {
            print(f"StartGenerateAdvanced failed: {e}");
            _update_job_state(
                self.projectName,
                "generate_adv",
                {"status": "error", "message": f"StartGenerateAdvanced failed: {e}"}
            );
            report {"success": False, "error": "StartGenerateAdvanced failed."};
        }
    }
}

walker:pub StartOptimizeAdvanced {
    has projectName: str,
        ops: list = [],
        iterations: int = 3,
        patience: int = 2,
        minImproveMs: float = 0.0,
        minSpeedupPct: float = 5.0,
        image: str = "cgins-codex:latest",
        targetDevice: str = "cuda";

    can run with `root entry {
        try {
            cfg = {};
            try {
                with open(config_path, "r") as f {
                    cfg = load(f);
                }
            } except Exception as e {
                print(f"Error loading config: {e}");
                cfg = config_store;
            }
            _apply_llm_env_from_config(cfg);

            target_device = (self.targetDevice or "cuda").lower();
            if target_device == "gpu" or target_device == "cuda" {
                target_device = "cuda";
            } elif target_device == "mps" {
                target_device = "mps";
            } elif target_device != "cpu" {
                target_device = "cuda";
            }
            os.environ["CGINS_TARGET_DEVICE"] = target_device;
            try {
                project_cfg_path = path.join(
                    _project_dir(self.projectName), "config.json"
                );
                project_cfg = {};
                if path.exists(project_cfg_path) {
                    with open(project_cfg_path, "r") as f {
                        project_cfg = load(f);
                    }
                }
                gen_cfg = project_cfg["generator"]
                if project_cfg and "generator" in project_cfg
                else {};
                gen_cfg["target_device"] = target_device;
                project_cfg["generator"] = gen_cfg;
                with open(project_cfg_path, "w") as f {
                    dump(project_cfg, f, indent=2);
                }
            } except Exception as e {
                print(f"Failed to persist target device: {e}");
            }

            ops_list = [];
            ops_total = 0;
            for o in self.ops {
                if o {
                    ops_list.append(str(o));
                    ops_total = ops_total + 1;
                }
            }
            if ops_total == 0 {
                _update_job_state(
                    self.projectName,
                    "optimize_adv",
                    {"status": "error", "message": "No ops selected or discovered."}
                );
                report {"success": False, "error": "No ops selected or discovered."};
                disengage;
            }
            io_root = path.join(_project_dir(self.projectName), "io", "individual_ops");
            missing_inputs = [];
            missing_total = 0;
            for op_name in ops_list {
                op_dir = path.join(io_root, str(op_name));
                has_entry = False;
                if path.exists(op_dir) {
                    for fname in listdir(op_dir) {
                        if fname.endswith(".pt") {
                            has_entry = True;
                            break;
                        }
                    }
                }
                if not has_entry {
                    missing_inputs.append(op_name);
                    missing_total = missing_total + 1;
                }
            }
            if missing_total > 0 {
                error_msg = (
                    f"Missing input entries for {missing_total} ops. " + "Run Profile first."
                );
                _update_job_state(
                    self.projectName,
                    "optimize_adv",
                    {"status": "error", "message": error_msg}
                );
                report {"success": False, "error": error_msg};
                disengage;
            }

            now_ts = datetime.now().timestamp();
            _update_job_state(
                self.projectName,
                "optimize_adv",
                {
                    "status": "running",
                    "started_at": now_ts,
                    "message": f"Queued advanced optimization ({target_device})",
                    "progress": {
                        "current": 0,
                        "total": ops_total,
                        "percent": 0.0,
                        "updated_at": now_ts
                    }
                }
            );

            ops_rel = path.join("projects", self.projectName, "ops_selected.json");
            ops_path = path.join(repo_root, ops_rel);
            try {
                with open(ops_path, "w") as f {
                    dump(ops_list, f, indent=2);
                }
            } except Exception as e {
                print(f"Failed to write ops list: {e}");
            }

            cmd = [
                "python3",
                "scripts/codex_advanced_generate.py",
                "--project",
                self.projectName,
                "--ops-file",
                ops_rel,
                "--optimize-only",
                "--optimize"
            ];
            if self.iterations > 0 {
                cmd.append("--opt-iterations");
                cmd.append(str(self.iterations));
            }
            if self.patience > 0 {
                cmd.append("--opt-patience");
                cmd.append(str(self.patience));
            }
            cmd.append("--opt-min-improve-ms");
            cmd.append(str(self.minImproveMs));
            cmd.append("--opt-min-speedup-pct");
            cmd.append(str(self.minSpeedupPct));
            result = _start_job_container(
                self.projectName, "optimize_adv", cmd, "optimize_adv.log", self.image
            );
            report result;
        } except Exception as e {
            print(f"StartOptimizeAdvanced failed: {e}");
            _update_job_state(
                self.projectName,
                "optimize_adv",
                {"status": "error", "message": f"StartOptimizeAdvanced failed: {e}"}
            );
            report {"success": False, "error": "StartOptimizeAdvanced failed."};
        }
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
        gen_logs = _list_files(gen_attempt_dir, [".txt", ".log", ".json"]);
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
cl import from "./context/ConfigContext.cl.jac" { ConfigProvider }
cl import from "./app/Home.cl.jac" { Home }
cl import from "./app/Project.cl.jac" { Project }
cl import from "./app/Settings.cl.jac" { Settings }
cl import from "./app/NewProject.cl.jac" { NewProject }
cl import from "./app/OpDetails.cl.jac" { OpDetails }
cl import "./styles.css";

cl {
    def NotFound {
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

    def:pub app {
        useEffect(lambda -> None { document.title = "CGinS";}, []);
        return
            <Router>
                <ConfigProvider>
                    <div className="min-h-screen font-sans antialiased">
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
