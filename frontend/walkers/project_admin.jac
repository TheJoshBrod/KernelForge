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
