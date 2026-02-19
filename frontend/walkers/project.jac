"""Project management walkers."""

import from json { load, loads, dump, dumps }
import from os { path, makedirs, listdir }
import from base64 { b64decode, b64encode }
import from datetime { datetime }
import from zipfile { ZipFile }
import sys;
import subprocess;
import os;
import sqlite3;
import hashlib;

import from .globals { repo_root, config_path, config_store }

# --- Helpers ---
def _project_dir(name: str) -> str {
    primary = path.join(_projects_base_dir(), name);
    legacy = path.join(repo_root, "projects", name);
    if path.exists(primary) {
        return primary;
    }
    if path.exists(legacy) {
        return legacy;
    }
    return primary;
}

def _ensure_dir(dir_path: str) -> None {
    if not path.exists(dir_path) {
        makedirs(dir_path);
    }
}

def _projects_base_dir() -> str {
    data_dir = (
        os.environ["CGINS_DATA_DIR"]
        if "CGINS_DATA_DIR" in os.environ and os.environ["CGINS_DATA_DIR"]
        else path.join(repo_root, "kernels")
    );
    projects_dir = path.join(data_dir, "projects");
    _ensure_dir(projects_dir);
    return projects_dir;
}

def _legacy_projects_dir() -> str {
    return path.join(repo_root, "projects");
}

def _all_projects_dirs() -> list {
    dirs = [];
    primary = _projects_base_dir();
    legacy = _legacy_projects_dir();
    dirs.append(primary);
    if legacy != primary and path.exists(legacy) {
        dirs.append(legacy);
    }
    return dirs;
}

def _ensure_legacy_project_link(project_name: str, target_dir: str) -> None {
    legacy_root = _legacy_projects_dir();
    _ensure_dir(legacy_root);
    legacy_path = path.join(legacy_root, project_name);
    if path.abspath(legacy_path) == path.abspath(target_dir) {
        return;
    }
    if path.exists(legacy_path) {
        return;
    }
    try {
        os.symlink(target_dir, legacy_path);
    } except Exception as e {
        print(f"Warning: unable to create legacy project link for {project_name}: {e}");
    }
}

def _store_weight_artifact(project_path: str, file_name: Any, payload_b64: Any) -> dict {
    if not payload_b64 {
        return {};
    }
    source_name = str(file_name) if file_name else "weights.pth";
    if not (source_name.endswith(".pth") or source_name.endswith(".pt")) {
        source_name = source_name + ".pth";
    }
    data = b64decode(str(payload_b64));
    digest = hashlib.sha256(data).hexdigest();
    ext = ".pth" if source_name.endswith(".pth") else ".pt";

    artifacts_root = path.join(project_path, "artifacts");
    weights_root = path.join(artifacts_root, "weights");
    _ensure_dir(artifacts_root);
    _ensure_dir(weights_root);

    stored_name = digest + ext;
    stored_path = path.join(weights_root, stored_name);
    if not path.exists(stored_path) {
        with open(stored_path, "wb") as f {
            f.write(data);
        }
    }

    return {
        "id": digest[0:16],
        "name": source_name,
        "sha256": digest,
        "size_bytes": len(data),
        "relpath": path.relpath(stored_path, project_path),
        "source": "upload"
    };
}

def _merge_weight_artifacts(entries: list) -> list {
    dedup = {};
    out = [];
    for item in entries {
        if not item or "sha256" not in item {
            continue;
        }
        key = item["sha256"];
        if key in dedup {
            continue;
        }
        dedup[key] = True;
        out.append(item);
    }
    return out;
}

def _apply_llm_env_from_config(
    cfg: dict,
    project_cfg: dict = {},
    override_provider: str = "",
    override_model: str = ""
) -> dict {
    if not cfg {
        return {};
    }
    try {
        code = """
import json
import sys
from src.llm.runtime_config import resolve_runtime_env

global_cfg = json.loads(sys.argv[1])
project_cfg = json.loads(sys.argv[2])
override_provider = sys.argv[3]
override_model = sys.argv[4]

env_map = resolve_runtime_env(
    global_config=global_cfg,
    project_config=project_cfg,
    override_provider=override_provider,
    override_model=override_model,
)
print(json.dumps(env_map))
""";
        out = subprocess.check_output(
            [
                sys.executable,
                "-c",
                code,
                dumps(cfg),
                dumps(project_cfg if project_cfg else {}),
                override_provider if override_provider else "",
                override_model if override_model else ""
            ],
            cwd=repo_root,
            timeout=8
        );
        text = out.decode("utf-8").strip();
        env_map = loads(text) if text else {};
        for key in env_map {
            value = env_map[key];
            if value is not None {
                os.environ[str(key)] = str(value);
            }
        }
        return env_map;
    } except Exception as e {
        print(f"Failed to apply LLM env: {e}");
        return {};
    }
}

def _read_state_raw(state_path: str) -> dict {
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

def _reconcile_legacy_profile_state(
    project_name: str, state: dict, state_path: str
) -> dict {
    if not state or not ("profile" in state) or not ("prepare" in state) {
        return state;
    }

    profile = state["profile"];
    prepare = state["prepare"];
    if not profile or not prepare {
        return state;
    }

    profile_status = (
        str(profile["status"]).strip().lower()
        if "status" in profile and profile["status"]
        else ""
    );
    prepare_status = (
        str(prepare["status"]).strip().lower()
        if "status" in prepare and prepare["status"]
        else ""
    );
    profile_message = (
        str(profile["message"]).strip().lower()
        if "message" in profile and profile["message"]
        else ""
    );

    if not (
        profile_status == "queued"
        and prepare_status == "completed"
        and profile_message == "preparing assets"
    ) {
        return state;
    }

    project_dir = _project_dir(project_name);
    bench_path = path.join(project_dir, "benchmarks", "op_benchmarks.json");
    prepare_log_path = path.join(project_dir, "logs", "prepare.log");
    recovery_started = (
        bool(profile["recovery_started"]) if "recovery_started" in profile else False
    );

    if path.exists(bench_path) {
        profile["status"] = "completed";
        profile["message"] = "Recovered from legacy prepare flow";
        profile["recovery_started"] = False;
    } else {
        if not recovery_started {
            cmd = [
                sys.executable,
                path.join(repo_root, "scripts", "profile_then_benchmark.py"),
                "--project",
                project_name
            ];
            result = _start_job(project_name, "profile", cmd, "profile.log");
            if result and "success" in result and result["success"] {
                profile["status"] = "running";
                profile["message"] = "Recovering legacy profile pipeline";
                profile["recovery_started"] = True;
            } else {
                profile["status"] = "error";
                profile["message"] = "Failed to recover legacy profile pipeline";
            }
        } else {
            err_msg = "Legacy profile recovery in progress. Check profile.log.";
            if path.exists(prepare_log_path) {
                log_text = "";
                try {
                    with open(prepare_log_path, "r") as f {
                        log_text = f.read();
                    }
                } except Exception as e {
                    log_text = "";
                }
                if "Profiling failed" in log_text or "Traceback" in log_text {
                    err_msg = "Profiling failed in legacy prepare flow. Check prepare.log.";
                }
            }
            if profile["status"] == "queued" {
                profile["status"] = "running";
            }
            profile["message"] = err_msg;
        }
    }

    state["profile"] = profile;
    try {
        with open(state_path, "w") as f {
            dump(state, f, indent=2);
        }
    } except Exception as e {
        print(f"Failed to write reconciled state.json: {e}");
    }
    return state;
}

def _read_state(project_name: str) -> dict {
    state_path = path.join(_project_dir(project_name), "state.json");
    state = _read_state_raw(state_path);
    return _reconcile_legacy_profile_state(project_name, state, state_path);
}

def _update_job_state(project_name: str, job_key: str, updates: dict) -> None {
    state_path = path.join(_project_dir(project_name), "state.json");
    state = _read_state_raw(state_path);
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
        code = "import json; from src.optimizer.profiling import get_frontend_payload; print(json.dumps(get_frontend_payload(mode='fast')))";
        out = subprocess.check_output([sys.executable, "-c", code], cwd=repo_root);
        text = out.decode("utf-8").strip();
        if text {
            return loads(text);
        }
    } except Exception as e {
        return {
            "available": False,
            "gpus": [],
            "source": "local",
            "cuda_available": False,
            "mps_available": False,
            "xpu_available": False,
            "preferred_target": "cpu",
            "error": str(e)
        };
    }
    return {
        "available": False,
        "gpus": [],
        "source": "local",
        "cuda_available": False,
        "mps_available": False,
        "xpu_available": False,
        "preferred_target": "cpu"
    };
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
        path.join("kernels", "projects", project_name),
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

# --- Walkers ---
# --- Walkers ---
walker:pub CreateProject {
    has projectName: str,
        code: str,
        modelFileB64: str = "",
        modelFileName: str = "model.py",
        weightsFileB64: str = "",
        weightsFileName: str = "",
        pthFiles: list = [],
        validationSetPath: str = "",
        validationZipB64: str = "",
        validationZipName: str = "";

    can create with entry {
        try {
            projects_dir = _projects_base_dir();
            _ensure_dir(projects_dir);

            project_path = path.join(projects_dir, self.projectName);
            legacy_path = path.join(_legacy_projects_dir(), self.projectName);

            if path.exists(project_path) or path.exists(legacy_path) {
                report {"error": "Project already exists", "success": False};
                disengage;
            }

            makedirs(project_path);
            _ensure_legacy_project_link(self.projectName, project_path);

            model_written = False;
            if self.code and self.code.strip() {
                with open(path.join(project_path, "model.py"), "w") as f {
                    f.write(self.code);
                }
                model_written = True;
            } elif self.modelFileB64 {
                model_bytes = b64decode(self.modelFileB64);
                with open(path.join(project_path, "model.py"), "wb") as f {
                    f.write(model_bytes);
                }
                model_written = True;
            }

            if not model_written {
                report {
                    "error": "Model code or model file is required",
                    "success": False
                };
                disengage;
            }

            weight_artifacts = [];
            if self.weightsFileB64 {
                weights_data = b64decode(self.weightsFileB64);
                with open(path.join(project_path, "weights.pt"), "wb") as f {
                    f.write(weights_data);
                }
                legacy_meta = _store_weight_artifact(
                    project_path,
                    (self.weightsFileName if self.weightsFileName else "weights.pt"),
                    self.weightsFileB64
                );
                if legacy_meta {
                    weight_artifacts.append(legacy_meta);
                }
            }

            if self.pthFiles {
                for item in self.pthFiles {
                    if not item {
                        continue;
                    }
                    p_name = str(item["name"])
                    if "name" in item and item["name"]
                    else "checkpoint.pth";
                    p_b64 = str(item["b64"]) if "b64" in item and item["b64"] else "";
                    if not p_b64 {
                        continue;
                    }
                    p_meta = _store_weight_artifact(project_path, p_name, p_b64);
                    if p_meta {
                        weight_artifacts.append(p_meta);
                    }
                }
            }
            weight_artifacts = _merge_weight_artifacts(weight_artifacts);

            validation_dir = self.validationSetPath;
            if self.validationZipB64 {
                validation_dir = path.join("data", "validation");
            }

            hw = _detect_hardware();
            target_device = "cpu";
            if hw and "preferred_target" in hw and hw["preferred_target"] {
                target_device = hw["preferred_target"];
            } elif hw and "mps_available" in hw and hw["mps_available"] {
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
                "profile": {"allow_ops": [], "skip_ops": [], "skip_prefixes": []},
                "artifacts": {"weights": weight_artifacts}
            };
            with open(path.join(project_path, "config.json"), "w") as f {
                dump(config, f, indent=2);
            }

            prepare_result = {};
            profile_result = {};
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
            }

            profile_cmd = [
                sys.executable,
                path.join(repo_root, "scripts", "profile_then_benchmark.py"),
                "--project",
                self.projectName
            ];
            if weights_b64_path {
                profile_cmd.append("--weights-b64-path");
                profile_cmd.append(weights_b64_path);
            }
            if validation_b64_path {
                profile_cmd.append("--validation-b64-path");
                profile_cmd.append(validation_b64_path);
            }
            if validation_name_path {
                profile_cmd.append("--validation-name-path");
                profile_cmd.append(validation_name_path);
            }
            profile_result = _start_job(
                self.projectName, "profile", profile_cmd, "profile.log"
            );

            report {
                "success": True,
                "name": self.projectName,
                "weights_count": len(weight_artifacts),
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
    can return_projects with entry {
        try {
            projects = [];
            seen = {};
            for projects_dir in _all_projects_dirs() {
                if not path.exists(projects_dir) {
                    continue;
                }
                for name in listdir(projects_dir) {
                    if name in seen {
                        continue;
                    }
                    full_path = path.join(projects_dir, name);
                    if path.isdir(full_path) {
                        seen[name] = True;
                        projects.append({"name": name});
                    }
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

    can return_ops with entry {
        try {
            base_dir = path.join(
                _project_dir(self.projectName), "io", "individual_ops"
            );
            ops_map = {};

            if path.exists(base_dir) {
                for name in listdir(base_dir) {
                    full_path = path.join(base_dir, name);
                    if path.isdir(full_path) {
                        count = 0;
                        # Quick count: list only entry files
                        try {
                            entries = [
                                f
                                for f in listdir(full_path)
                                if f.startswith("entry_") and f.endswith(".pt")
                            ];
                            count = len(entries);
                        } except Exception {
                            count = 0;
                        }
                        ops_map[name] = count;
                    }
                }
            }

            if len(ops_map) == 0 {
                summary_path = path.join(
                    _project_dir(self.projectName), "io", "summary.json"
                );
                if path.exists(summary_path) {
                    try {
                        with open(summary_path, "r") as f {
                            summary = load(f);
                        }
                        op_counts = summary["op_counts"]
                        if summary and "op_counts" in summary
                        else {};
                        for full_name in op_counts {
                            op_name = full_name.replace(".", "_").replace("/", "_");
                            c = op_counts[full_name];
                            try {
                                c = int(c);
                            } except Exception {
                                c = 0;
                            }
                            ops_map[op_name] = c;
                        }
                    } except Exception as e {
                        print(f"Error reading summary for ops: {e}");
                    }
                }
            }

            ops = [];
            for name in ops_map {
                ops.append({"name": name, "count": ops_map[name]});
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

    can return_status with entry {
        state = _read_state(self.projectName);
        report state;
    }
}

walker:pub SetJobControl {
    has projectName: str,
        jobKey: str,
        action: str;

    can run with entry {
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

    can run with entry {
        cmd = [
            sys.executable,
            path.join(repo_root, "scripts", "profile_then_benchmark.py"),
            "--project",
            self.projectName
        ];
        result = _start_job(self.projectName, "profile", cmd, "profile.log");
        report result;
    }
}

walker:pub StartGenerate {
    has projectName: str;

    can run with entry {
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
        optimize: bool = True,
        benchmark: bool = True,
        iterations: int = 0,
        patience: int = 0,
        genAttempts: int = 3,
        minImproveMs: float = 0.0,
        minSpeedupPct: float = 5.0,
        image: str = "cgins-codex:latest",
        llm_provider: str = "",
        llm_model: str = "",
        generator_config: dict = {},
        optimizer_config: dict = {};

    can run with entry {
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

            project_cfg_path = path.join(_project_dir(self.projectName), "config.json");
            project_cfg = {};
            try {
                if path.exists(project_cfg_path) {
                    with open(project_cfg_path, "r") as f {
                        project_cfg = load(f);
                    }
                }
            } except Exception as e {
                print(f"Failed to load project config: {e}");
                project_cfg = {};
            }

            llm_cfg = project_cfg["llm"]
            if project_cfg and "llm" in project_cfg
            else {};
            if self.llm_provider {
                llm_cfg["provider"] = str(self.llm_provider).strip().lower();
            }
            if self.llm_model {
                llm_cfg["model"] = str(self.llm_model).strip();
            }
            if llm_cfg {
                project_cfg["llm"] = llm_cfg;
            }
            os.environ["CGINS_PROJECT_CONFIG_PATH"] = project_cfg_path;
            _apply_llm_env_from_config(
                cfg,
                project_cfg,
                self.llm_provider if self.llm_provider else "",
                self.llm_model if self.llm_model else ""
            );

            # Extract generator config overrides
            target_device = "cuda";
            if self.generator_config and "target_device" in self.generator_config {
                target_device = self.generator_config["target_device"];
            }

            target_device = target_device.lower();
            if target_device == "gpu"
            or target_device == "cuda"
            or target_device == "rocm"
            or target_device == "amd" {
                target_device = "cuda";
            } elif target_device == "mps"
            or target_device == "metal"
            or target_device == "apple" {
                target_device = "mps";
            } elif target_device == "xpu" or target_device == "intel" {
                target_device = "cpu";
            } elif target_device != "cpu" {
                target_device = "cpu";
            }
            os.environ["CGINS_TARGET_DEVICE"] = target_device;

            # Persist project config overrides if needed
            try {
                # Update generator config
                gen_cfg = project_cfg["generator"]
                if project_cfg and "generator" in project_cfg
                else {};
                gen_cfg["target_device"] = target_device;

                if self.generator_config {
                    # Apply other generator overrides
                    if "max_attempts" in self.generator_config {
                        gen_cfg["max_attempts"] = self.generator_config["max_attempts"];
                    }
                    if "use_baseline_kernels" in self.generator_config {
                        gen_cfg["use_baseline_kernels"] = self.generator_config[
                            "use_baseline_kernels"
                        ];
                    }
                    if "use_baseline_as_template" in self.generator_config {
                        gen_cfg["use_baseline_as_template"] = self.generator_config[
                            "use_baseline_as_template"
                        ];
                    }
                    if "extra_validation_cases" in self.generator_config {
                        gen_cfg["extra_validation_cases"] = self.generator_config[
                            "extra_validation_cases"
                        ];
                    }
                }
                project_cfg["generator"] = gen_cfg;

                with open(project_cfg_path, "w") as f {
                    dump(project_cfg, f, indent=2);
                }
            } except Exception as e {
                print(f"Failed to persist config overrides: {e}");
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
            if self.generator_config and "max_attempts" in self.generator_config {
                cmd.append("--generate-attempts");
                cmd.append(str(self.generator_config["max_attempts"]));
            }

            if self.llm_model {
                cmd.append("--codex-model");
                cmd.append(self.llm_model);
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
        llm_provider: str = "",
        llm_model: str = "",
        optimizer_config: dict = {};

    can run with entry {
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
            project_cfg_path = path.join(_project_dir(self.projectName), "config.json");
            project_cfg = {};
            try {
                if path.exists(project_cfg_path) {
                    with open(project_cfg_path, "r") as f {
                        project_cfg = load(f);
                    }
                }
            } except Exception as e {
                print(f"Failed to load project config: {e}");
                project_cfg = {};
            }

            llm_cfg = project_cfg["llm"]
            if project_cfg and "llm" in project_cfg
            else {};
            llm_changed = False;
            if self.llm_provider {
                llm_cfg["provider"] = str(self.llm_provider).strip().lower();
                llm_changed = True;
            }
            if self.llm_model {
                llm_cfg["model"] = str(self.llm_model).strip();
                llm_changed = True;
            }
            if llm_cfg {
                project_cfg["llm"] = llm_cfg;
            }
            if llm_changed {
                try {
                    with open(project_cfg_path, "w") as f {
                        dump(project_cfg, f, indent=2);
                    }
                } except Exception as e {
                    print(f"Failed to persist project LLM config: {e}");
                }
            }
            os.environ["CGINS_PROJECT_CONFIG_PATH"] = project_cfg_path;
            _apply_llm_env_from_config(
                cfg,
                project_cfg,
                self.llm_provider if self.llm_provider else "",
                self.llm_model if self.llm_model else ""
            );

            # Load target device from project config
            target_device = "cuda";
            if project_cfg
            and "generator" in project_cfg
            and "target_device" in project_cfg["generator"] {
                target_device = project_cfg["generator"]["target_device"];
            }
            target_device = target_device.lower();
            if target_device == "gpu"
            or target_device == "cuda"
            or target_device == "rocm"
            or target_device == "amd" {
                target_device = "cuda";
            } elif target_device == "mps"
            or target_device == "metal"
            or target_device == "apple" {
                target_device = "mps";
            } elif target_device == "xpu" or target_device == "intel" {
                target_device = "cpu";
            } elif target_device != "cpu" {
                target_device = "cpu";
            }
            os.environ["CGINS_TARGET_DEVICE"] = target_device;

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

            if self.llm_model {
                cmd.append("--codex-model");
                cmd.append(self.llm_model);
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

    can run with entry {
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

    can run with entry {
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

    can run with entry {
        result = _export_project_zip(self.projectName, self.ops);
        report result;
    }
}

walker:pub GetOpDetails {
    has projectName: str,
        opName: str;

    can run with entry {
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

    can run with entry {
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

    can run with entry {
        if self.ops and len(self.ops) > 0 {
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

    can run with entry {
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

walker:pub GetMctsTree {
    has projectName: str,
        opName: str;

    can run with entry {
        project_dir = _project_dir(self.projectName);
        db_path = path.join(project_dir, "trees", self.opName, "nodes.db");

        if not path.exists(db_path) {
            report {
                "success": True,
                "nodes": [],
                "edges": [],
                "message": "No optimization tree yet for this operator."
            };
            disengage;
        }

        mcts_nodes = [];
        edges = [];

        try {
            # Since Jac can interop with Python, we can use sqlite3 directly
            conn = sqlite3.connect(db_path);
            # Set row factory to dict-like if needed, but manual extraction is safer in Jac logic
            cursor = conn.cursor();

            # Fetch nodes
            cursor.execute(
                "SELECT id, visits, value, best_subtree_value, improvement_description, timestamp, code FROM nodes"
            );
            rows = cursor.fetchall();
            for row in rows {
                mcts_nodes.append(
                    {
                        "id": row[0],
                        "visits": row[1],
                        "value": row[2],
                        "best_subtree_value": row[3],
                        "improvement_description": row[4],
                        "timestamp": row[5],
                        "code": row[6]
                    }
                );
            }

            # Fetch edges
            cursor.execute("SELECT parent_id, child_id FROM edges");
            rows = cursor.fetchall();
            for row in rows {
                edges.append({"parent_id": row[0], "child_id": row[1]});
            }

            conn.close();
            report {"success": True, "nodes": mcts_nodes, "edges": edges};
        } except Exception as e {
            report {"success": False, "error": str(e)};
        }
    }
}

walker:pub ImportKernel {
    can run with entry {
        print("upload!");
        report {"success": True, "message": "upload!"};
    }
}

walker:pub GetProjectMctsSummary {
    has projectName: str;

    can run with entry {
        project_dir = _project_dir(self.projectName);
        trees_dir = path.join(project_dir, "trees");

        total_nodes = 0;
        op_summaries = [];

        if path.exists(trees_dir) {
            for op_name in listdir(trees_dir) {
                db_path = path.join(trees_dir, op_name, "nodes.db");
                if path.exists(db_path) {
                    try {
                        conn = sqlite3.connect(db_path);
                        cursor = conn.cursor();

                        cursor.execute("SELECT COUNT(*) FROM nodes");
                        node_count = cursor.fetchone()[0];

                        cursor.execute("SELECT MAX(value) FROM nodes");
                        best_val = cursor.fetchone()[0] or 0.0;

                        total_nodes += node_count;
                        op_summaries.append(
                            {
                                "op": op_name,
                                "nodes": node_count,
                                "best_value": best_val
                            }
                        );
                        conn.close();
                    } except Exception as e {
                        print(f"Error summarying {op_name}: {e}");
                    }
                }
            }
        }

        report {"success": True, "total_nodes": total_nodes, "ops": op_summaries};
    }
}

walker:pub GetDashboardCharts {
    has projectName: str;

    can return_charts with entry {
        project_dir = _project_dir(self.projectName);

        bench_results = [];
        bench_file_status = "";
        bench_errors = [];
        bench_path = path.join(project_dir, "benchmarks", "op_benchmarks.json");
        if path.exists(bench_path) {
            try {
                with open(bench_path, "r") as f {
                    bench_data = load(f);
                }
                bench_results = bench_data["results"]
                if bench_data and "results" in bench_data
                else [];
                bench_file_status = (
                    str(bench_data["status"])
                    if bench_data and "status" in bench_data and bench_data["status"]
                    else ""
                );
                bench_errors = (
                    bench_data["errors"]
                    if bench_data and "errors" in bench_data and bench_data["errors"]
                    else []
                );
            } except Exception as e {
                print(f"Error reading op_benchmarks.json: {e}");
                bench_results = [];
                bench_file_status = "error";
                bench_errors = [str(e)];
            }
        }

        op_counts = {};
        op_profile_ms = {};
        summary_path = path.join(project_dir, "io", "summary.json");
        if path.exists(summary_path) {
            try {
                with open(summary_path, "r") as f {
                    summary_data = load(f);
                }
                op_counts = summary_data["op_counts"]
                if summary_data and "op_counts" in summary_data
                else {};
                op_profile_ms = summary_data["op_profile_ms"]
                if summary_data and "op_profile_ms" in summary_data
                else {};
            } except Exception as e {
                print(f"Error reading io/summary.json: {e}");
                op_counts = {};
                op_profile_ms = {};
            }
        }

        # Convert counts to the on-disk op directory naming convention:
        # "torch.nn.functional.linear" -> "torch_nn_functional_linear"
        count_by_op = {};
        for full_name in op_counts {
            try {
                op_dir_name = full_name.replace(".", "_").replace("/", "_");
                count_by_op[op_dir_name] = op_counts[full_name];
            } except Exception {
                continue;
            }
        }
        profile_ms_by_op = {};
        for full_name in op_profile_ms {
            try {
                op_dir_name = full_name.replace(".", "_").replace("/", "_");
                ms_val = float(op_profile_ms[full_name]);
                profile_ms_by_op[op_dir_name] = ms_val;
            } except Exception {
                continue;
            }
        }

        bench_by_op = {};
        for r in bench_results {
            if r and "op" in r and r["op"] {
                bench_by_op[r["op"]] = r;
            }
        }

        speed_comparison = [];
        for r in bench_results {
            if not r or not ("op" in r) {
                continue;
            }
            op_name = r["op"];
            pytorch_ms = 0.0;
            if r and "pytorch_ms" in r and r["pytorch_ms"] is not None {
                try {
                    pytorch_ms = float(r["pytorch_ms"]);
                } except Exception {
                    pytorch_ms = 0.0;
                }
            }

            row = {"name": op_name, "pytorch": pytorch_ms};

            if (
                r
                and "kernel_status" in r
                and r["kernel_status"] == "ok"
                and "kernel_ms" in r
                and r["kernel_ms"] is not None
            ) {
                optimized_ms = None;
                try {
                    optimized_ms = float(r["kernel_ms"]);
                } except Exception {
                    optimized_ms = None;
                }
                if optimized_ms is not None {
                    row["optimized"] = optimized_ms;
                }
            }
            speed_comparison.append(row);
        }
        for op_name in profile_ms_by_op {
            if op_name in bench_by_op {
                continue;
            }
            ms_val = profile_ms_by_op[op_name];
            if ms_val and ms_val > 0.0 {
                speed_comparison.append({"name": op_name, "pytorch": ms_val});
            }
        }

        operator_usage = [];
        all_ops = [];
        seen_ops = {};
        for op_name in bench_by_op {
            if not (op_name in seen_ops) {
                all_ops.append(op_name);
                seen_ops[op_name] = True;
            }
        }
        for op_name in count_by_op {
            if not (op_name in seen_ops) {
                all_ops.append(op_name);
                seen_ops[op_name] = True;
            }
        }
        for op_name in profile_ms_by_op {
            if not (op_name in seen_ops) {
                all_ops.append(op_name);
                seen_ops[op_name] = True;
            }
        }

        for op_name in all_ops {
            pytorch_ms = 0.0;
            b = {};
            if op_name in bench_by_op {
                b = bench_by_op[op_name];
            }
            if op_name in bench_by_op {
                if b and "pytorch_ms" in b and b["pytorch_ms"] is not None {
                    try {
                        pytorch_ms = float(b["pytorch_ms"]);
                    } except Exception {
                        pytorch_ms = 0.0;
                    }
                }
            } elif op_name in profile_ms_by_op {
                pytorch_ms = profile_ms_by_op[op_name];
                if pytorch_ms < 0.0 {
                    pytorch_ms = 0.0;
                }
            }

            count = count_by_op[op_name] if op_name in count_by_op else 0;
            if count <= 0 and b and "entries" in b and b["entries"] {
                count = b["entries"];
            }
            try {
                count = int(count);
            } except Exception {
                count = 0;
            }

            total_time_ms = pytorch_ms * count if pytorch_ms and count else 0.0;
            if total_time_ms > 0.0 {
                operator_usage.append(
                    {"name": op_name, "count": count, "total_time_ms": total_time_ms}
                );
            }
        }

        state = _read_state(self.projectName);
        profile_status = "";
        profile_message = "";
        if state and "profile" in state and state["profile"] {
            p = state["profile"];
            profile_status = str(p["status"]) if "status" in p and p["status"] else "";
            profile_message = str(p["message"])
            if "message" in p and p["message"]
            else "";
        }

        chart_status = "ready";
        if len(speed_comparison) == 0 and len(operator_usage) == 0 {
            p_status = profile_status.lower() if profile_status else "";
            if bench_file_status == "partial" {
                chart_status = "partial";
            } elif bench_file_status and bench_file_status != "ready" {
                chart_status = bench_file_status;
            } elif p_status == "running"
            or p_status == "queued"
            or p_status == "paused" {
                chart_status = "pending";
            } elif p_status == "error" {
                chart_status = "error";
            } else {
                chart_status = "empty";
            }
        }

        report {
            "operator_usage": operator_usage,
            "speed_comparison": speed_comparison,
            "status": chart_status,
            "profile_status": profile_status,
            "profile_message": profile_message,
            "errors": bench_errors
        };
    }
}
