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

import from .globals { repo_root, config_store }
import from .catalog_db {
    _ensure_dir,
    _projects_base_dir,
    _catalog_db_path,
    _ensure_catalog_schema,
    _upsert_project_catalog,
    _load_project_catalog,
    _touch_project_last_opened
}
import from .job_supervisor {
    _project_dir,
    _all_projects_dirs,
    _migrate_legacy_projects,
    _detect_hardware,
    _get_active_project_job,
    _queue_has_backlog,
    _queue_has_running_work,
    _start_job,
    _find_optimized_dir,
    _normalize_metric_op_name,
    _read_state,
    _read_queue_state,
    _start_next_queued_job
}

# --- Private helpers ---

def _short_gpu_name(gpu_info: Any) -> str {
    if not gpu_info {
        return "";
    }
    name = str(gpu_info["name"]) if "name" in gpu_info and gpu_info["name"] else "";
    if not name {
        return "";
    }
    prefixes = [
        "NVIDIA GeForce ",
        "NVIDIA Quadro ",
        "NVIDIA Tesla ",
        "AMD Radeon ",
        "Intel Arc "
    ];
    for prefix in prefixes {
        if name.startswith(prefix) {
            return name[len(prefix):];
        }
    }
    return name;
}

def _check_gpu_mismatch(project_name: str, current_gpu_info: Any) -> dict {
    cfg_path = path.join(_project_dir(project_name), "config.json");
    if not path.exists(cfg_path) {
        return {"gpu_mismatch": False};
    }
    try {
        with open(cfg_path, "r") as f {
            cfg = load(f);
        }
    } except Exception {
        return {"gpu_mismatch": False};
    }
    if not cfg or "benchmarked_gpu" not in cfg {
        return {"gpu_mismatch": False};
    }
    benchmarked = cfg["benchmarked_gpu"];
    stored_uuid = str(benchmarked["uuid"]) if "uuid" in benchmarked and benchmarked["uuid"] else "";
    stored_name = str(benchmarked["name"]) if "name" in benchmarked and benchmarked["name"] else "Unknown";
    stored_conn = str(benchmarked["connection"]) if "connection" in benchmarked and benchmarked["connection"] else "local";
    if not current_gpu_info {
        return {"gpu_mismatch": False};
    }
    cur_uuid = str(current_gpu_info["uuid"]) if "uuid" in current_gpu_info and current_gpu_info["uuid"] else "";
    cur_name = str(current_gpu_info["name"]) if "name" in current_gpu_info and current_gpu_info["name"] else "Unknown";
    cur_conn = str(current_gpu_info["connection"]) if "connection" in current_gpu_info and current_gpu_info["connection"] else "local";
    if stored_uuid and cur_uuid {
        is_match = stored_uuid == cur_uuid;
    } else {
        is_match = stored_name == cur_name and stored_conn == cur_conn;
    }
    if not is_match {
        return {
            "gpu_mismatch": True,
            "benchmarked_gpu_name": stored_name,
            "current_gpu_name": cur_name
        };
    }
    return {"gpu_mismatch": False};
}

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
        selectedGpuInfo: dict = {},
        selectedBackend: str = "",
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

            base_name = self.projectName;
            gpu_info = self.selectedGpuInfo if self.selectedGpuInfo else config_store["selected_gpu_info"];
            short_gpu = _short_gpu_name(gpu_info);
            final_name = f"{base_name} - {short_gpu}" if short_gpu else base_name;

            project_path = path.join(projects_dir, final_name);
            if path.exists(project_path) {
                report {"error": "Project already exists", "success": False};
                disengage;
            }

            makedirs(project_path);

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

            benchmarked_gpu_entry = None;
            if gpu_info {
                bg_uuid = str(gpu_info["uuid"]) if "uuid" in gpu_info and gpu_info["uuid"] else "";
                bg_name = str(gpu_info["name"]) if "name" in gpu_info and gpu_info["name"] else "Unknown";
                bg_conn = str(gpu_info["connection"]) if "connection" in gpu_info and gpu_info["connection"] else "local";
                benchmarked_gpu_entry = {
                    "uuid": bg_uuid,
                    "name": bg_name,
                    "connection": bg_conn,
                    "identity": bg_uuid if bg_uuid else f"{bg_name}:{bg_conn}"
                };
            }
            config = {
                "base_name": base_name,
                "validation_dir": validation_dir,
                "created_at": datetime.now().isoformat(),
                "backend": self.selectedBackend if self.selectedBackend else target_device,
                "generator": {
                    "skip_ops": [],
                    "only_ops": [],
                    "max_ops": 0,
                    "use_baseline_kernels": True,
                    "use_baseline_as_template": True,
                    "max_attempts": 8,
                    "extra_validation_cases": 1,
                    "target_device": self.selectedBackend if self.selectedBackend else target_device
                },
                "profile": {"allow_ops": [], "skip_ops": [], "skip_prefixes": []},
                "artifacts": {"weights": weight_artifacts}
            };
            if benchmarked_gpu_entry {
                config["benchmarked_gpu"] = benchmarked_gpu_entry;
            }
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
                final_name
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
                final_name, "profile", profile_cmd, "profile.log"
            );
            _upsert_project_catalog(final_name, project_path);

            report {
                "success": True,
                "name": final_name,
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

walker:pub TouchProject {
    has projectName: str;

    can touch_project with entry {
        try {
            if self.projectName {
                _touch_project_last_opened(self.projectName);
            }
            report {"ok": True};
        } except Exception as e {
            print(f"TouchProject error: {e}");
            report {"ok": False};
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

            # Primary source: summary.json op_counts (single forward pass counts)
            summary_path = path.join(
                _project_dir(self.projectName), "io", "summary.json"
            );
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

            # Fallback: count entry files if summary.json has nothing
            if len(ops_map) == 0 {
                try {
                    for name in listdir(base_dir) {
                        full_path = path.join(base_dir, name);
                        if path.isdir(full_path) {
                            count = 0;
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
                } except Exception {
                    ops_map = {};
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
    has projectName: str,
        currentGpuInfo: dict = {};

    can return_status with entry {
        _start_next_queued_job(self.projectName);
        state = _read_state(self.projectName);
        queue_state = _read_queue_state(self.projectName);
        if (
            not _get_active_project_job(self.projectName, False)
            and _queue_has_backlog(queue_state)
            and not _queue_has_running_work(queue_state)
        ) {
            _start_next_queued_job(self.projectName);
            state = _read_state(self.projectName);
            queue_state = _read_queue_state(self.projectName);
        }
        if "active_tasks" in queue_state {
            cleaned = {};
            for (k, v) in queue_state["active_tasks"].items() {
                if v {
                    cleaned[k] = v;
                }
            }
            queue_state["active_tasks"] = cleaned;
        }
        state["queue"] = queue_state;
        current_gpu_info = self.currentGpuInfo if self.currentGpuInfo else config_store["selected_gpu_info"];
        mismatch_info = _check_gpu_mismatch(self.projectName, current_gpu_info);
        state["gpu_mismatch"] = mismatch_info["gpu_mismatch"] if "gpu_mismatch" in mismatch_info else False;
        if "benchmarked_gpu_name" in mismatch_info {
            state["benchmarked_gpu_name"] = mismatch_info["benchmarked_gpu_name"];
        }
        if "current_gpu_name" in mismatch_info {
            state["current_gpu_name"] = mismatch_info["current_gpu_name"];
        }

        # Retrieve backend from config
        cfg_path = path.join(_project_dir(self.projectName), "config.json");
        config_data = {};
        if path.exists(cfg_path) {
            try {
                with open(cfg_path, "r") as f {
                    config_data = load(f);
                }
            } except Exception as e {
                print(f"Failed to read project config: {e}");
            }
        }
        state["backend"] = config_data["backend"] if "backend" in config_data else (config_data["generator"]["target_device"] if "generator" in config_data and "target_device" in config_data["generator"] else "cuda");

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

walker:pub CloneProjectForGpu {
    has sourceProjectName: str,
        currentGpuInfo: dict = {};

    can run with entry {
        try {
            src_dir = _project_dir(self.sourceProjectName);
            src_cfg_path = path.join(src_dir, "config.json");
            base_name = self.sourceProjectName;
            src_cfg = {};
            if path.exists(src_cfg_path) {
                try {
                    with open(src_cfg_path, "r") as f {
                        src_cfg = load(f);
                    }
                } except Exception {
                    src_cfg = {};
                }
            }
            if src_cfg and "base_name" in src_cfg and src_cfg["base_name"] {
                base_name = str(src_cfg["base_name"]);
            } else {
                last_dash = self.sourceProjectName.rfind(" - ");
                if last_dash != -1 {
                    base_name = self.sourceProjectName[:last_dash];
                }
            }

            gpu_info = self.currentGpuInfo if self.currentGpuInfo else config_store["selected_gpu_info"];
            short_name = _short_gpu_name(gpu_info);
            new_name = f"{base_name} - {short_name}" if short_name else base_name;

            projects_dir = _projects_base_dir();
            if path.exists(path.join(projects_dir, new_name)) {
                new_name = f"{new_name}_2";
            }

            new_dir = path.join(projects_dir, new_name);
            makedirs(new_dir);

            src_trees = path.join(src_dir, "trees");
            new_trees = path.join(new_dir, "trees");
            if path.exists(src_trees) {
                for op_name in listdir(src_trees) {
                    src_op = path.join(src_trees, op_name);
                    if not path.isdir(src_op) {
                        continue;
                    }
                    new_op = path.join(new_trees, op_name);
                    _ensure_dir(new_op);
                    src_kernels = path.join(src_op, "kernels");
                    if path.exists(src_kernels) {
                        shutil.copytree(src_kernels, path.join(new_op, "kernels"));
                    }
                }
            }

            src_gen = path.join(src_dir, "kernels", "generated");
            if path.exists(src_gen) {
                shutil.copytree(src_gen, path.join(new_dir, "kernels", "generated"));
            }

            src_io = path.join(src_dir, "io");
            if path.exists(src_io) {
                shutil.copytree(src_io, path.join(new_dir, "io"));
            }

            src_model = path.join(src_dir, "model.py");
            if path.exists(src_model) {
                shutil.copy2(src_model, path.join(new_dir, "model.py"));
            }

            src_artifacts = path.join(src_dir, "artifacts");
            if path.exists(src_artifacts) {
                shutil.copytree(src_artifacts, path.join(new_dir, "artifacts"));
            }

            new_cfg = {**src_cfg} if src_cfg else {};
            new_cfg["base_name"] = base_name;
            new_cfg["created_at"] = datetime.now().isoformat();
            if "benchmarked_gpu" in new_cfg {
                new_cfg.pop("benchmarked_gpu");
            }
            if gpu_info {
                clone_uuid = str(gpu_info["uuid"]) if "uuid" in gpu_info and gpu_info["uuid"] else "";
                clone_name = str(gpu_info["name"]) if "name" in gpu_info and gpu_info["name"] else "Unknown";
                clone_conn = str(gpu_info["connection"]) if "connection" in gpu_info and gpu_info["connection"] else "local";
                new_cfg["benchmarked_gpu"] = {
                    "uuid": clone_uuid,
                    "name": clone_name,
                    "connection": clone_conn,
                    "identity": clone_uuid if clone_uuid else f"{clone_name}:{clone_conn}"
                };
            }
            with open(path.join(new_dir, "config.json"), "w") as f {
                dump(new_cfg, f, indent=2);
            }

            _upsert_project_catalog(new_name, new_dir);

            bench_cmd = [
                sys.executable,
                "-m",
                "src.optimizer.benchmarking.benchmark_ops",
                "--project",
                new_name
            ];
            _start_job(new_name, "benchmark", bench_cmd, "benchmark.log");

            report {"success": True, "newProjectName": new_name};
        } except Exception as e {
            print(f"Error cloning project: {e}");
            report {"success": False, "error": str(e)};
        }
    }
}

walker:pub GetEnrichedDAG {
    has projectName: str;

    can run with entry {
        try {
            proj_dir = _project_dir(self.projectName);

            # Read dag.json
            dag_nodes = [];
            dag_edges = [];
            try {
                dag_path = path.join(proj_dir, "io", "dag.json");
                with open(dag_path, "r") as f {
                    dag_data = load(f);
                }
                dag_nodes = dag_data["nodes"] if dag_data and "nodes" in dag_data else [];
                dag_edges = dag_data["edges"] if dag_data and "edges" in dag_data else [];
            } except Exception as e {
                print(f"GetEnrichedDAG: dag.json: {e}");
            }

            # Read summary.json for single-pass call counts
            count_map = {};
            try {
                summary_path = path.join(proj_dir, "io", "summary.json");
                with open(summary_path, "r") as f {
                    summary = load(f);
                }
                op_counts = summary["op_counts"] if summary and "op_counts" in summary else {};
                for full_name in op_counts {
                    normalized = _normalize_metric_op_name(full_name);
                    short = normalized[len("torch_nn_functional_"):] if normalized.startswith("torch_nn_functional_") else normalized;
                    try {
                        count_map[short] = int(op_counts[full_name]);
                    } except Exception {
                        count_map[short] = 0;
                    }
                }
            } except Exception as e {
                print(f"GetEnrichedDAG: summary.json: {e}");
            }

            # Read op_benchmarks.json
            bench_map = {};
            bench_op_for = {};
            try {
                bench_path = path.join(proj_dir, "benchmarks", "op_benchmarks.json");
                with open(bench_path, "r") as f {
                    bench_data = load(f);
                }
                bench_results = bench_data["results"] if bench_data and "results" in bench_data else [];
                for r in bench_results {
                    if not r {
                        continue;
                    }
                    bench_op = r["op"] if "op" in r and r["op"] else "";
                    if not bench_op {
                        continue;
                    }
                    short = bench_op[len("torch_nn_functional_"):] if bench_op.startswith("torch_nn_functional_") else bench_op;
                    bench_map[short] = r;
                    bench_op_for[short] = bench_op;
                }
            } except Exception as e {
                print(f"GetEnrichedDAG: op_benchmarks.json: {e}");
            }

            # Read fusion_groups.json (optional)
            fusion_groups = [];
            try {
                fg_path = path.join(proj_dir, "fusion_groups.json");
                with open(fg_path, "r") as f {
                    fg_data = load(f);
                }
                fusion_groups = fg_data["groups"] if fg_data and "groups" in fg_data else [];
            } except Exception {
                fusion_groups = [];
            }

            # Build enriched nodes
            total_pytorch_ms = 0.0;
            enriched_nodes = [];
            for n in dag_nodes {
                node_id = n["id"] if "id" in n and n["id"] else "";
                op = n["op"] if "op" in n and n["op"] else "";
                if not node_id or not op {
                    continue;
                }

                call_count = count_map[op] if op in count_map else 0;
                bench = bench_map[op] if op in bench_map else {};
                full_bench_op = bench_op_for[op] if op in bench_op_for else f"torch_nn_functional_{op}";

                pytorch_ms = None;
                kernel_ms = None;
                winner = "";
                speedup = None;
                status = "pending";

                if bench {
                    pm = bench.get("pytorch_ms");
                    km = bench.get("kernel_ms");
                    wn = bench.get("winner", "");
                    sp = bench.get("speedup");
                    if pm is not None {
                        try { pytorch_ms = float(pm); } except Exception { }
                    }
                    if km is not None {
                        try { kernel_ms = float(km); } except Exception { }
                    }
                    if sp is not None {
                        try { speedup = float(sp); } except Exception { }
                    }
                    winner = str(wn) if wn else "";
                    if winner == "optimized" {
                        status = "forged";
                    } elif pytorch_ms is not None {
                        status = "benchmarked";
                    }
                }

                if pytorch_ms is not None {
                    total_pytorch_ms = total_pytorch_ms + pytorch_ms;
                }

                enriched_nodes.append({
                    "id": node_id,
                    "op": op,
                    "bench_op": full_bench_op,
                    "call_count": call_count,
                    "pytorch_ms": pytorch_ms,
                    "kernel_ms": kernel_ms,
                    "winner": winner,
                    "speedup": speedup,
                    "status": status
                });
            }

            report {
                "nodes": enriched_nodes,
                "edges": dag_edges,
                "fusion_groups": fusion_groups,
                "total_pytorch_ms": total_pytorch_ms
            };
        } except Exception as e {
            print(f"GetEnrichedDAG error: {e}");
            report {"nodes": [], "edges": [], "fusion_groups": [], "total_pytorch_ms": 0.0};
        }
    }
}

walker:pub SaveFusionGroups {
    has projectName: str,
        groups: list = [];

    can run with entry {
        try {
            proj_dir = _project_dir(self.projectName);
            fg_path = path.join(proj_dir, "fusion_groups.json");
            fg_data = {"version": 1, "groups": self.groups};
            with open(fg_path, "w") as f {
                dump(fg_data, f, indent=2);
            }
            report {"success": True};
        } except Exception as e {
            print(f"SaveFusionGroups error: {e}");
            report {"success": False, "error": str(e)};
        }
    }
}

walker:pub UpdateFusionGroup {
    has projectName: str,
        groupId: str,
        action: str,
        payload: dict = {};

    can run with entry {
        try {
            proj_dir = _project_dir(self.projectName);
            fg_path = path.join(proj_dir, "fusion_groups.json");

            try {
                with open(fg_path, "r") as f {
                    fg_data = load(f);
                }
            } except Exception {
                fg_data = {"version": 1, "groups": []};
            }

            groups = fg_data["groups"] if fg_data and "groups" in fg_data else [];

            # Find the target group to get its pattern_name
            target_pattern = None;
            for g in groups {
                if g.get("id") == self.groupId {
                    target_pattern = g.get("pattern_name", g.get("id"));
                    break;
                }
            }

            if target_pattern is not None {
                for g in groups {
                    g_pname = g.get("pattern_name", g.get("id"));
                    if g_pname == target_pattern {
                        if self.action == "accept" {
                            g["status"] = "accepted";
                        } elif self.action == "reject" {
                            g["status"] = "rejected";
                        } elif self.action == "rename" {
                            new_name = self.payload.get("name", "");
                            if new_name {
                                g["name"] = new_name;
                            }
                        }
                    }
                }
            }

            fg_data["groups"] = groups;
            with open(fg_path, "w") as f {
                dump(fg_data, f, indent=2);
            }

            report {"success": True, "groups": groups};
        } except Exception as e {
            print(f"UpdateFusionGroup error: {e}");
            report {"success": False, "error": str(e)};
        }
    }
}

walker:pub ProposeFusions {
    has projectName: str,
        model: str = "claude-sonnet-4-5-20251001";

    can run with entry {
        import json;
        import from .globals { config_path }

        try {
            proj_dir = _project_dir(self.projectName);

            # Load API keys
            api_keys = {};
            try {
                with open(config_path, "r") as f {
                    cfg = json.load(f);
                }
                api_keys = cfg["api_keys"] if cfg and "api_keys" in cfg else {};
            } except Exception as e {
                print(f"ProposeFusions: config error: {e}");
            }

            # Read dag.json
            dag_nodes = [];
            dag_edges = [];
            try {
                dag_path = path.join(proj_dir, "io", "dag.json");
                with open(dag_path, "r") as f {
                    dag_data = load(f);
                }
                dag_nodes = dag_data["nodes"] if dag_data and "nodes" in dag_data else [];
                dag_edges = dag_data["edges"] if dag_data and "edges" in dag_data else [];
            } except Exception as e {
                report {"success": False, "error": f"No DAG found — run profiling first. ({e})"};
                return;
            }

            if not dag_nodes {
                report {"success": False, "error": "DAG is empty — run profiling first."};
                return;
            }

            # Read op_benchmarks.json for timing context
            bench_map = {};
            try {
                bench_path = path.join(proj_dir, "benchmarks", "op_benchmarks.json");
                with open(bench_path, "r") as f {
                    bench_data = load(f);
                }
                bench_results = bench_data["results"] if bench_data and "results" in bench_data else [];
                for r in bench_results {
                    if not r {
                        continue;
                    }
                    bench_op = r["op"] if "op" in r and r["op"] else "";
                    if not bench_op {
                        continue;
                    }
                    short = bench_op[len("torch_nn_functional_"):] if bench_op.startswith("torch_nn_functional_") else bench_op;
                    bench_map[short] = r;
                }
            } except Exception as e {
                print(f"ProposeFusions: benchmarks: {e}");
            }

            # Build node op map and adjacency
            node_ops = {};
            for n in dag_nodes {
                nid = n["id"] if "id" in n else "";
                op = n["op"] if "op" in n else "";
                if nid and op {
                    node_ops[nid] = op;
                }
            }

            adj_out = {};
            adj_in = {};
            for edge in dag_edges {
                src = edge["source"] if "source" in edge else "";
                tgt = edge["target"] if "target" in edge else "";
                if not src or not tgt {
                    continue;
                }
                adj_out.setdefault(src, []).append(tgt);
                adj_in.setdefault(tgt, []).append(src);
            }

            # Walk linear chains. For each unvisited node, walk backward first to find
            # the true chain head (so iteration order doesn't split chains), then walk
            # forward to collect the full sequence.
            visited = set();
            all_chains = [];
            for n in dag_nodes {
                nid = n["id"] if "id" in n else "";
                if not nid or nid in visited {
                    continue;
                }

                # Backward walk — find chain head.
                # Uses a local set: global visited is still empty for this chain's nodes,
                # so it can't guard against cycles here on its own.
                head = nid;
                backward_visited = {nid};
                while True {
                    preds = adj_in.get(head, []);
                    if len(preds) != 1 {
                        break;
                    }
                    pred = preds[0];
                    if len(adj_out.get(pred, [])) != 1 {
                        break;
                    }
                    if pred in visited or pred in backward_visited {
                        break;
                    }
                    backward_visited.add(pred);
                    head = pred;
                }

                # Forward walk from head
                chain = [head];
                visited.add(head);
                current = head;
                while True {
                    outs = adj_out.get(current, []);
                    if len(outs) != 1 {
                        break;
                    }
                    next_id = outs[0];
                    if len(adj_in.get(next_id, [])) != 1 {
                        break;
                    }
                    if next_id in visited {
                        break;
                    }
                    chain.append(next_id);
                    visited.add(next_id);
                    current = next_id;
                }

                if len(chain) >= 2 {
                    all_chains.append(chain);
                }
            }

            # Group chains by op sequence
            pattern_groups = {};
            for chain in all_chains {
                if len(chain) < 2 {
                    continue;
                }
                op_seq_list = [];
                for nid in chain {
                    op_seq_list.append(node_ops.get(nid, "?"));
                }
                key = "|".join(op_seq_list);
                if key not in pattern_groups {
                    pattern_groups[key] = {"ops": op_seq_list, "instances": []};
                }
                pattern_groups[key]["instances"].append(chain);
            }

            # Sort patterns by frequency and compute per-pattern timing
            sorted_patterns = sorted(
                pattern_groups.items(),
                key=lambda x : -len(x[1]["instances"])
            );

            # Attach timing to each pattern entry so we can use it during expansion
            for (key, pg) in sorted_patterns {
                combined_ms = 0.0;
                for op in pg["ops"] {
                    pm = bench_map.get(op, {}).get("pytorch_ms");
                    if pm {
                        try { combined_ms = combined_ms + float(pm); } except Exception { combined_ms = combined_ms; }
                    }
                }
                pg["combined_ms"] = combined_ms;
            }

            # Build the candidate list sent to the LLM (top 20 by frequency)
            pattern_lines = [];
            for (key, pg) in sorted_patterns[:20] {
                ops = pg["ops"];
                freq = len(pg["instances"]);
                ms_str = f"{pg['combined_ms']:.3f}ms" if pg["combined_ms"] > 0 else "N/A";
                pattern_lines.append(f"  {key}: x{freq} — {ms_str}/instance");
            }

            op_timing_lines = [];
            for short_op in sorted(bench_map.keys())[:30] {
                r = bench_map[short_op];
                pm = r.get("pytorch_ms");
                km = r.get("kernel_ms");
                winner = r.get("winner", "");
                if pm {
                    status_tag = " [FORGED]" if winner == "optimized" else "";
                    km_str = f" / kernel={km:.3f}ms" if km else "";
                    op_timing_lines.append(f"  {short_op}: pytorch={pm:.3f}ms{km_str}{status_tag}");
                }
            }

            pattern_text = "\n".join(pattern_lines) if pattern_lines else "  (no repeated patterns)";
            op_timing_text = "\n".join(op_timing_lines) if op_timing_lines else "  (no benchmark data)";
            total_nodes = len(dag_nodes);

            system_prompt = """You are a GPU kernel fusion expert. The algorithm has already identified every valid linear chain in the compute graph. Your job is to RANK them — not select them. Every pattern will be shown to the user; your scores determine sort order and help them prioritize.

Return ONLY valid JSON (no markdown fences, no text outside JSON) in this exact schema:
{
  "rankings": [
    {
      "pattern_key": "op1|op2|op3",
      "score": 0.95,
      "estimated_speedup": 1.8,
      "rationale": "one sentence: why this fusion is or isn't worth doing"
    }
  ]
}

Scoring rules:
- score is 0.0–1.0; higher = stronger fusion candidate
- Score high (0.8–1.0): memory-bound chains where fusing eliminates global memory round-trips (conv+bn+relu, linear+gelu, linear+relu)
- Score medium (0.4–0.7): chains with mixed compute/memory profile or uncertain benefit
- Score low (0.0–0.3): already-[FORGED] chains, compute-bound ops unlikely to benefit, or very short chains with high launch overhead relative to compute
- estimated_speedup is the realistic multiplier assuming the fusion is implemented well (1.0 = no gain)
- You MUST return an entry for every pattern key listed — do not drop any""";

            user_prompt = f"""Project: {self.projectName}
Total ops in graph: {total_nodes}

Candidate fusion patterns (key: op|op|... → frequency → combined timing/instance):
{pattern_text}

Per-op baseline timing:
{op_timing_text}

Score every pattern listed above.""";

            # Call LLM
            model_name = self.model;
            response_text = "";

            if model_name.startswith("claude") {
                import anthropic;
                api_key = api_keys.get("anthropic", "");
                if not api_key {
                    report {"success": False, "error": "No Anthropic API key configured"};
                    return;
                }
                ant_client = anthropic.Anthropic(api_key=api_key);
                ant_msg = ant_client.messages.create(
                    model=model_name,
                    max_tokens=2048,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                );
                response_text = ant_msg.content[0].text;
            } elif model_name.startswith("gpt") or model_name.startswith("o1") or model_name.startswith("o3") or model_name.startswith("o4") {
                import openai;
                api_key = api_keys.get("openai", "");
                if not api_key {
                    report {"success": False, "error": "No OpenAI API key configured"};
                    return;
                }
                oai_client = openai.OpenAI(api_key=api_key);
                oai_resp = oai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=2048
                );
                response_text = oai_resp.choices[0].message.content;
            } elif model_name.startswith("gemini") {
                import from google { genai }
                api_key = api_keys.get("google", "");
                if not api_key {
                    report {"success": False, "error": "No Google API key configured"};
                    return;
                }
                gem_client = genai.Client(api_key=api_key);
                gem_resp = gem_client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config={"system_instruction": system_prompt}
                );
                response_text = gem_resp.text;
            } else {
                report {"success": False, "error": f"Unknown model provider for: {model_name}"};
                return;
            }

            # Parse rankings from LLM response
            parsed = None;
            try {
                text = response_text.strip();
                if text.startswith("```") {
                    lines = text.split("\n");
                    inner = lines[1:] if len(lines) > 1 else [];
                    if inner and inner[-1].strip().startswith("```") {
                        inner = inner[:-1];
                    }
                    text = "\n".join(inner);
                }
                parsed = json.loads(text);
            } except Exception as e {
                report {"success": False, "error": f"LLM response parse error: {e}\nRaw: {response_text[:500]}"};
                return;
            }

            rankings = parsed.get("rankings", []) if parsed else [];

            # Build a score lookup keyed by pattern_key
            score_map = {};
            for r in rankings {
                pk = r.get("pattern_key", "");
                if pk {
                    score_map[pk] = {
                        "score": float(r.get("score", 0.5)),
                        "estimated_speedup": float(r.get("estimated_speedup", 1.0)),
                        "rationale": r.get("rationale", "")
                    };
                }
            }

            # Expand ALL discovered patterns to instance-level groups, sorted by LLM score.
            # The LLM ranks; the user decides what to accept.
            scored_patterns = sorted(
                sorted_patterns,
                key=lambda x : -score_map.get(x[0], {}).get("score", 0.0)
            );

            new_groups = [];
            group_idx = 0;
            color_idx = 0;
            for (key, pg) in scored_patterns {
                ops = pg["ops"];
                instances = pg["instances"];
                scoring = score_map.get(key, {});
                score = scoring.get("score", 0.5);
                est_speedup = scoring.get("estimated_speedup", 1.0);
                rationale = scoring.get("rationale", "");
                pattern_name = "_".join(ops);

                inst_idx = 0;
                for instance in instances {
                    new_groups.append({
                        "id": f"fg_{group_idx}",
                        "pattern_name": pattern_name,
                        "name": f"{pattern_name}_{inst_idx}",
                        "members": instance,
                        "status": "proposed",
                        "score": score,
                        "estimated_speedup": est_speedup,
                        "rationale": rationale,
                        "color_index": color_idx % 8
                    });
                    group_idx = group_idx + 1;
                    inst_idx = inst_idx + 1;
                }
                color_idx = color_idx + 1;
            }

            # Save fusion_groups.json
            fg_path = path.join(proj_dir, "fusion_groups.json");
            fg_save = {"version": 1, "model_used": model_name, "groups": new_groups};
            with open(fg_path, "w") as f {
                dump(fg_save, f, indent=2);
            }

            report {"success": True, "groups": new_groups};
        } except Exception as e {
            print(f"ProposeFusions error: {e}");
            report {"success": False, "error": str(e)};
        }
    }
}
