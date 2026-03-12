"""Project CRUD walkers and file-handling helpers."""

import from json { load, dump }
import from os { path, makedirs, listdir }
import from base64 { b64decode, b64encode }
import from datetime { datetime }
import from zipfile { ZipFile }
import sys;
import os;
import shutil;
import hashlib;
import sqlite3;

import from .globals { repo_root }
import from .catalog_db {
    _ensure_dir,
    _projects_base_dir,
    _catalog_db_path,
    _ensure_catalog_schema,
    _upsert_project_catalog,
    _load_project_catalog
}
import from .job_supervisor {
    _project_dir,
    _all_projects_dirs,
    _migrate_legacy_projects,
    _detect_hardware,
    _start_job,
    _find_optimized_dir,
    _normalize_metric_op_name,
    _read_state
}

# --- Private helpers ---

def _prune_missing_catalog_projects() -> None {
    _ensure_catalog_schema();
    db_path = _catalog_db_path();
    conn = sqlite3.connect(db_path);
    cursor = conn.cursor();
    cursor.execute("SELECT name, project_path FROM projects");
    rows = cursor.fetchall();
    for row in rows {
        if not row or len(row) < 1 or not row[0] {
            continue;
        }
        p_name = str(row[0]);
        p_path = str(row[1]) if len(row) > 1 and row[1] else "";
        resolved_path = p_path if p_path else _project_dir(p_name);
        if path.exists(resolved_path) {
            continue;
        }
        cursor.execute("DELETE FROM projects WHERE name = ?", (p_name,));
        cursor.execute(
            "DELETE FROM operator_metrics WHERE project_name = ?",
            (p_name,)
        );
        cursor.execute(
            "DELETE FROM project_metric_meta WHERE project_name = ?",
            (p_name,)
        );
        cursor.execute("DELETE FROM run_state WHERE project_name = ?", (p_name,));
    }
    conn.commit();
    conn.close();
}

def _sync_project_catalog_from_fs() -> None {
    for projects_dir in _all_projects_dirs() {
        if not path.exists(projects_dir) {
            continue;
        }
        for name in listdir(projects_dir) {
            full_path = path.join(projects_dir, name);
            if not path.isdir(full_path) {
                continue;
            }
            _upsert_project_catalog(name, full_path);
        }
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
    export_path = path.join(export_dir, project_name + ".kforge.zip");

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
            _migrate_legacy_projects();
            projects_dir = _projects_base_dir();
            _ensure_dir(projects_dir);

            project_path = path.join(projects_dir, self.projectName);
            if path.exists(project_path) {
                report {"error": "Project already exists", "success": False};
                disengage;
            }

            makedirs(project_path);

            # === TEMPORARY: HARDCODED MODEL FIX FOR LOCAL TESTING ===
            temp_model_path = "/mnt/c/Users/lsgre/Downloads/model/model/restnet-50.py";
            temp_weights_path = "/mnt/c/Users/lsgre/Downloads/model/model/weights.pth";

            model_written = False;
            if path.exists(temp_model_path) {
                shutil.copyfile(temp_model_path, path.join(project_path, "model.py"));
                model_written = True;
                print("DEBUG: Copied hardcoded model file");
            } elif self.code and self.code.strip() {
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
                    "error": "Model code or model file is required (hardcoded file not found)",
                    "success": False
                };
                disengage;
            }

            weight_artifacts = [];
            if path.exists(temp_weights_path) {
                shutil.copyfile(temp_weights_path, path.join(project_path, "weights.pt"));
                print("DEBUG: Copied hardcoded weights file");
                legacy_meta = _store_weight_artifact(
                    project_path,
                    "weights.pth",
                    "" # No base64 data to store
                );
                if legacy_meta {
                    weight_artifacts.append(legacy_meta);
                }
            } elif self.weightsFileB64 {
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
            # =======================================================

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
                "-m",
                "src.optimizer.workflow",
                "profile",
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
            _upsert_project_catalog(self.projectName, project_path);

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
            _sync_project_catalog_from_fs();
            _prune_missing_catalog_projects();
            projects = [];
            catalog_rows = _load_project_catalog();
            for row in catalog_rows {
                if not row or not ("name" in row) or not row["name"] {
                    continue;
                }
                p_name = str(row["name"]);
                p_path = (
                    str(row["project_path"])
                    if "project_path" in row and row["project_path"]
                    else _project_dir(p_name)
                );
                if not path.exists(p_path) or not path.isdir(p_path) {
                    continue;
                }
                projects.append({"name": p_name});
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
                            op_name = _normalize_metric_op_name(full_name);
                            if not op_name {
                                continue;
                            }
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

            # Read per-call pytorch_ms from op_benchmarks.json
            profile_ms_map = {};
            bench_path = path.join(
                _project_dir(self.projectName), "benchmarks", "op_benchmarks.json"
            );
            if path.exists(bench_path) {
                try {
                    with open(bench_path, "r") as bf {
                        bench_data = load(bf);
                    }
                    bench_results = (
                        bench_data["results"]
                        if bench_data and "results" in bench_data
                        else []
                    );
                    for r in bench_results {
                        if not r {
                            continue;
                        }
                        op_name = r["op"] if "op" in r and r["op"] else "";
                        if not op_name {
                            continue;
                        }
                        ms_val = r["pytorch_ms"] if "pytorch_ms" in r and r["pytorch_ms"] is not None else None;
                        if ms_val is not None {
                            try {
                                profile_ms_map[op_name] = float(ms_val);
                            } except Exception {
                                # no-op
                            }
                        }
                    }
                } except Exception as e {
                    print(f"Error reading op_benchmarks.json: {e}");
                }
            }

            ops = [];
            for name in ops_map {
                ms = profile_ms_map[name] if name in profile_ms_map else 0.0;
                ops.append({"name": name, "count": ops_map[name], "profile_ms": ms});
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
        project_dir = _project_dir(self.projectName);
        queue_path = path.join(project_dir, "queue.json");
        queue_state = {};
        lock_path = queue_path + ".lock";
        lock_fd = None;
        try {
            import fcntl;
            lock_fd = open(lock_path, "a+");
            fcntl.flock(lock_fd, fcntl.LOCK_SH);
        } except Exception {
            lock_fd = None;
        }
        try {
            if path.exists(queue_path) {
                try {
                    with open(queue_path, "r") as f {
                        queue_state = load(f);
                    }
                } except Exception as e {
                    print(f"Failed to read queue state: {e}");
                }
            }
        } finally {
            if lock_fd {
                try {
                    import fcntl;
                    fcntl.flock(lock_fd, fcntl.LOCK_UN);
                } except Exception {
                    # no-op
                }
                lock_fd.close();
            }
        }
        # Note: we no longer filter out "Starting" entries — that filtering
        # caused tasks to disappear from the UI during the handoff window.
        if "active_tasks" in queue_state {
            if len(queue_state["active_tasks"]) == 0 {
                queue_state["current_operator"] = "";
                queue_state["pending_operators"] = [];
            }
        }
        # Scan for generated ops (success.* files) and failed ops (failure.json) — persists across clears
        generated_ops = [];
        failed_ops = [];
        gen_kernels_dir = path.join(
            project_dir, "kernels", "generated", "individual_op_kernels"
        );
        if path.exists(gen_kernels_dir) {
            try {
                for op_dir_name in listdir(gen_kernels_dir) {
                    op_dir_path = path.join(gen_kernels_dir, op_dir_name);
                    if not path.isdir(op_dir_path) {
                        continue;
                    }
                    has_success = False;
                    for fname in listdir(op_dir_path) {
                        if fname.startswith("success.") {
                            generated_ops.append(op_dir_name);
                            has_success = True;
                            break;
                        }
                    }
                    if not has_success {
                        attempts_dir = path.join(op_dir_path, "attempts");
                        if path.exists(attempts_dir) and path.exists(path.join(attempts_dir, "failure.json")) {
                            failed_ops.append(op_dir_name);
                        }
                    }
                }
            } except Exception {
                # no-op
            }
        }
        state["queue"] = queue_state;
        state["generated_ops"] = generated_ops;
        state["failed_ops"] = failed_ops;
        report state;
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

walker:pub ExportProject {
    has projectName: str,
        ops: list = [];

    can run with entry {
        result = _export_project_zip(self.projectName, self.ops);
        report result;
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
                self.projectName + ".kforge.zip"
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

walker:pub DeleteProject {
    has projectName: str;

    can execute with entry {
        try {
            proj_dir = _project_dir(self.projectName);
            if path.exists(proj_dir) {
                shutil.rmtree(proj_dir);
                report {"success": True, "message": f"Project {self.projectName} deleted successfully"};
            } else {
                report {"success": False, "message": f"Project {self.projectName} not found"};
            }
        } except Exception as e {
            report {"success": False, "message": f"Error deleting project: {e}"};
        }
    }
}
