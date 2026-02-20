"""Project management walkers."""

import from json { load, loads, dump, dumps }
import from os { path, makedirs, listdir }
import from base64 { b64decode, b64encode }
import from datetime { datetime }
import from zipfile { ZipFile }
import sys;
import subprocess;
import os;
import shutil;
import sqlite3;
import hashlib;

import from .globals { repo_root, config_path, config_store }

# --- Helpers ---
def _project_dir(name: str) -> str {
    _migrate_legacy_projects();
    return path.join(_projects_base_dir(), name);
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

def _migrate_legacy_projects() -> None {
    projects_dir = _projects_base_dir();
    legacy_roots = [
        path.join(repo_root, "kernels", "private", "projects"),
        path.join(repo_root, "projects")
    ];
    for legacy_root in legacy_roots {
        if not path.exists(legacy_root) {
            continue;
        }
        for name in listdir(legacy_root) {
            src_path = path.join(legacy_root, name);
            dst_path = path.join(projects_dir, name);
            if path.exists(dst_path) {
                continue;
            }
            source = src_path;
            try {
                if path.islink(src_path) {
                    source = path.realpath(src_path);
                    if not path.exists(source) or not path.isdir(source) {
                        continue;
                    }
                }
                os.rename(source, dst_path);
                if path.islink(src_path) and path.exists(src_path) {
                    os.unlink(src_path);
                }
            } except Exception as e {
                try {
                    if path.exists(source) and path.isdir(source) {
                        shutil.copytree(source, dst_path, dirs_exist_ok=True);
                        if source == src_path and path.exists(src_path) {
                            shutil.rmtree(src_path);
                        }
                    }
                } except Exception as copy_err {
                    print(f"Warning: failed to migrate project {name}: {copy_err}");
                }
            }
        }
        try {
            if len(listdir(legacy_root)) == 0 {
                os.rmdir(legacy_root);
            }
        } except Exception as e {
            _ = str(e);
        }
    }
}

def _all_projects_dirs() -> list {
    _migrate_legacy_projects();
    return [_projects_base_dir()];
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
        profile["control"] = "idle";
        profile["active"] = False;
        profile["phase"] = "done";
        profile["progress_percent"] = 1.0;
        profile["message"] = "Recovered from legacy prepare flow";
        profile["recovery_started"] = False;
    } else {
        if not recovery_started {
            cmd = [
                sys.executable,
                "-m",
                "src.optimizer.benchmarking.pipeline",
                "--project",
                project_name
            ];
            result = _start_job(project_name, "profile", cmd, "profile.log");
            if result and "success" in result and result["success"] {
                profile["status"] = "running";
                profile["control"] = "running";
                profile["active"] = True;
                profile["phase"] = "profiling";
                profile["progress_percent"] = 0.35;
                profile["message"] = "Recovering legacy profile pipeline";
                profile["recovery_started"] = True;
            } else {
                profile["status"] = "error";
                profile["control"] = "idle";
                profile["active"] = False;
                profile["phase"] = "error";
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
            profile["control"] = "running";
            profile["active"] = True;
            profile["phase"] = "profiling";
            profile["message"] = err_msg;
        }
    }

    profile_status = (
        str(profile["status"]).strip().lower()
        if "status" in profile and profile["status"]
        else ""
    );
    if (
        profile_status == "completed"
        or profile_status == "success"
        or profile_status == "cancelled"
        or profile_status == "error"
    ) {
        profile["control"] = "idle";
        profile["active"] = False;
        if not ("phase" in profile) or not profile["phase"] {
            profile["phase"] = (
                "done"
                if profile_status == "completed" or profile_status == "success"
                else "error"
            );
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

def _normalize_job_flags(project_name: str, state: dict, state_path: str) -> dict {
    if not state {
        return {};
    }
    changed = False;
    if "generate_adv" in state {
        if not ("generate" in state) {
            state["generate"] = state["generate_adv"];
        }
        state.pop("generate_adv", None);
        changed = True;
    }
    for job_key in state {
        job = state[job_key];
        if not job or not ("status" in job) {
            continue;
        }
        status_raw = str(job["status"]).strip().lower() if job["status"] else "";
        control_raw = (
            str(job["control"]).strip().lower()
            if "control" in job and job["control"]
            else ""
        );
        pid_val = job["pid"] if "pid" in job else None;
        has_live_pid = _pid_is_alive(pid_val);

        if (
            status_raw == "running" or status_raw == "queued" or status_raw == "paused"
        )
        and has_live_pid {
            if _refresh_running_job_state(project_name, job_key, job) {
                changed = True;
            }
            continue;
        }

        if (
            status_raw == "completed"
            or status_raw == "success"
            or status_raw == "error"
            or status_raw == "cancelled"
        )
        and (control_raw == "running" or ("active" in job and job["active"])) {
            job["control"] = "idle";
            job["active"] = False;
            if not ("phase" in job) or not job["phase"] {
                job["phase"] = (
                    "done"
                    if status_raw == "completed" or status_raw == "success"
                    else "error"
                );
            }
            if (status_raw == "completed" or status_raw == "success")
            and (not ("progress_percent" in job) or job["progress_percent"] is None) {
                job["progress_percent"] = 1.0;
            }
            changed = True;
            continue;
        }

        if (
            status_raw == "running" or status_raw == "queued" or status_raw == "paused"
        )
        and not has_live_pid {
            started_at = (
                str(job["started_at"])
                if "started_at" in job and job["started_at"]
                else ""
            );
            progress_raw = job["progress"] if "progress" in job and job["progress"] else {};
            progress_total = 0;
            progress_current = 0;
            try {
                progress_total = int(progress_raw["total"]) if "total" in progress_raw else 0;
            } except Exception {
                progress_total = 0;
            }
            try {
                progress_current = int(progress_raw["current"])
                if "current" in progress_raw
                else 0;
            } except Exception {
                progress_current = 0;
            }
            progress_complete = progress_total > 0 and progress_current >= progress_total;
            message_text = (
                str(job["message"]).strip().lower()
                if "message" in job and job["message"]
                else ""
            );
            nonfatal_generate_summary = (
                "unable to forge" in message_text
                or ("forged " in message_text and "operators" in message_text)
            );
            if job_key == "profile" and _has_profile_outputs(project_name) {
                job["status"] = "completed";
                job["control"] = "idle";
                job["active"] = False;
                job["phase"] = "done";
                job["progress_percent"] = 1.0;
                if not ("message" in job) or not job["message"] {
                    job["message"] = "Profile and baseline benchmark already available.";
                }
            } elif job_key == "generate"
            and (
                (
                    progress_complete
                    and (
                        _has_generate_outputs_after(project_name, started_at)
                        or _has_optimize_outputs_after(project_name, started_at)
                        or _has_benchmark_outputs_after(project_name, started_at)
                        or nonfatal_generate_summary
                    )
                )
                or (
                    progress_total <= 0
                    and (
                        _has_generate_outputs_after(project_name, started_at)
                        or _has_optimize_outputs_after(project_name, started_at)
                        or _has_benchmark_outputs_after(project_name, started_at)
                    )
                )
            ) {
                job["status"] = "completed";
                job["control"] = "idle";
                job["active"] = False;
                job["phase"] = "done";
                job["progress_percent"] = 1.0;
                if _has_benchmark_outputs_after(project_name, started_at) {
                    if not ("message" in job) or not job["message"] {
                        job["message"] = "Kernel generation and benchmark completed.";
                    }
                } elif _has_optimize_outputs_after(project_name, started_at) {
                    if not ("message" in job) or not job["message"] {
                        job["message"] = "Kernel generation and optimization completed.";
                    }
                } elif not ("message" in job) or not job["message"] {
                    job["message"] = "Kernel generation completed.";
                }
            } elif job_key == "optimize"
            and _has_optimize_outputs_after(project_name, started_at) {
                job["status"] = "completed";
                job["control"] = "idle";
                job["active"] = False;
                job["phase"] = "done";
                job["progress_percent"] = 1.0;
                if not ("message" in job) or not job["message"] {
                    job["message"] = "Kernel optimization completed.";
                }
            } elif job_key == "benchmark"
            and _has_benchmark_outputs_after(project_name, started_at) {
                job["status"] = "completed";
                job["control"] = "idle";
                job["active"] = False;
                job["phase"] = "done";
                job["progress_percent"] = 1.0;
                if not ("message" in job) or not job["message"] {
                    job["message"] = "Benchmark completed.";
                }
            } else {
                job["status"] = "error";
                job["control"] = "idle";
                job["active"] = False;
                job["phase"] = "error";
                error_summary = "";
                if "log" in job and job["log"] {
                    error_summary = _extract_job_error_summary(str(job["log"]));
                }
                if error_summary {
                    job["message"] = error_summary;
                } else {
                    job["message"] = "Recovered stale job state (process not running).";
                }
            }
            changed = True;
        }
    }

    if changed {
        try {
            with open(state_path, "w") as f {
                dump(state, f, indent=2);
            }
        } except Exception as e {
            print(f"Failed to write normalized state.json: {e}");
        }
    }
    return state;
}

def _read_state(project_name: str) -> dict {
    state_path = path.join(_project_dir(project_name), "state.json");
    state = _read_state_raw(state_path);
    state = _reconcile_legacy_profile_state(project_name, state, state_path);
    state = _normalize_job_flags(project_name, state, state_path);
    if state and len(state) > 0 {
        _write_project_run_state_db(project_name, state);
    }
    db_state = _read_project_run_state_db(project_name);
    if db_state and len(db_state) > 0 {
        db_state = _normalize_job_flags(project_name, db_state, state_path);
        _write_project_run_state_db(project_name, db_state);
        return db_state;
    }
    return state if state else {};
}

def _update_job_state(project_name: str, job_key: str, updates: dict) -> None {
    state_path = path.join(_project_dir(project_name), "state.json");
    state = _read_state_raw(state_path);
    if not state or len(state) == 0 {
        state = _read_project_run_state_db(project_name);
    }
    job_state = state[job_key] if job_key in state else {};
    for k in updates {
        job_state[k] = updates[k];
    }
    if not ("updated_at" in job_state) or not job_state["updated_at"] {
        job_state["updated_at"] = datetime.now().isoformat();
    }
    state[job_key] = job_state;
    with open(state_path, "w") as f {
        dump(state, f, indent=2);
    }
    _write_job_state_db(project_name, job_key, job_state);
    _upsert_project_catalog(project_name, _project_dir(project_name));
}

def _pid_is_alive(pid_val: Any) -> bool {
    if pid_val is None {
        return False;
    }
    pid = 0;
    try {
        pid = int(pid_val);
    } except Exception {
        return False;
    }
    if pid <= 0 {
        return False;
    }
    try {
        proc_stat = path.join("/proc", str(pid), "stat");
        if path.exists(proc_stat) {
            with open(proc_stat, "r") as f {
                raw = f.read().strip();
            }
            close_idx = raw.rfind(")");
            if close_idx >= 0 and close_idx + 2 < len(raw) {
                proc_state = raw[close_idx + 2];
                if proc_state == "Z" {
                    return False;
                }
            }
        }
    } except Exception as e {
        _ = str(e);
    }
    try {
        os.kill(pid, 0);
        return True;
    } except Exception {
        return False;
    }
}

def _has_profile_outputs(project_name: str) -> bool {
    project_dir = _project_dir(project_name);
    bench_path = path.join(project_dir, "benchmarks", "op_benchmarks.json");
    summary_path = path.join(project_dir, "io", "summary.json");
    return path.exists(bench_path) and path.exists(summary_path);
}

def _has_generate_outputs(project_name: str) -> bool {
    generated_root = path.join(
        _project_dir(project_name), "kernels", "generated", "individual_op_kernels"
    );
    if not path.exists(generated_root) or not path.isdir(generated_root) {
        return False;
    }
    for op_name in listdir(generated_root) {
        op_dir = path.join(generated_root, op_name);
        if not path.isdir(op_dir) {
            continue;
        }
        if path.exists(path.join(op_dir, "success.cuda"))
        or path.exists(path.join(op_dir, "success.triton"))
        or path.exists(path.join(op_dir, "success.mps"))
        or path.exists(path.join(op_dir, "success.cpu")) {
            return True;
        }
    }
    return False;
}

def _iso_to_epoch(iso_value: str) -> float {
    if not iso_value {
        return 0.0;
    }
    try {
        text = str(iso_value);
        if text.endswith("Z") {
            text = text[0 : len(text) - 1] + "+00:00";
        }
        return datetime.fromisoformat(text).timestamp();
    } except Exception {
        return 0.0;
    }
}

def _modified_after(file_path: str, started_at: str) -> bool {
    if not path.exists(file_path) {
        return False;
    }
    started_ts = _iso_to_epoch(started_at);
    if started_ts <= 0.0 {
        return True;
    }
    try {
        return path.getmtime(file_path) >= (started_ts - 1.0);
    } except Exception {
        return False;
    }
}

def _has_generate_outputs_after(project_name: str, started_at: str = "") -> bool {
    generated_root = path.join(
        _project_dir(project_name), "kernels", "generated", "individual_op_kernels"
    );
    if not path.exists(generated_root) or not path.isdir(generated_root) {
        return False;
    }
    for op_name in listdir(generated_root) {
        op_dir = path.join(generated_root, op_name);
        if not path.isdir(op_dir) {
            continue;
        }
        markers = [
            path.join(op_dir, "success.cuda"),
            path.join(op_dir, "success.triton"),
            path.join(op_dir, "success.mps"),
            path.join(op_dir, "success.cpu")
        ];
        for marker in markers {
            if _modified_after(marker, started_at) {
                return True;
            }
        }
    }
    return False;
}

def _has_optimize_outputs(project_name: str) -> bool {
    trees_dir = path.join(_project_dir(project_name), "trees");
    if not path.exists(trees_dir) or not path.isdir(trees_dir) {
        return False;
    }
    for op_name in listdir(trees_dir) {
        db_path = path.join(trees_dir, op_name, "nodes.db");
        if path.exists(db_path) {
            return True;
        }
    }
    return False;
}

def _has_optimize_outputs_after(project_name: str, started_at: str = "") -> bool {
    trees_dir = path.join(_project_dir(project_name), "trees");
    if not path.exists(trees_dir) or not path.isdir(trees_dir) {
        return False;
    }
    for op_name in listdir(trees_dir) {
        db_path = path.join(trees_dir, op_name, "nodes.db");
        if _modified_after(db_path, started_at) {
            return True;
        }
    }
    return False;
}

def _load_tree_best_by_op(project_name: str) -> dict {
    best_by_op = {};
    trees_dir = path.join(_project_dir(project_name), "trees");
    if not path.exists(trees_dir) or not path.isdir(trees_dir) {
        return best_by_op;
    }
    for op_name in listdir(trees_dir) {
        db_path = path.join(trees_dir, op_name, "nodes.db");
        if not path.exists(db_path) {
            continue;
        }
        try {
            conn = sqlite3.connect(db_path);
            cursor = conn.cursor();
            cursor.execute("SELECT MIN(value) FROM nodes WHERE value IS NOT NULL AND value > 0");
            row = cursor.fetchone();
            conn.close();
            if row and row[0] is not None {
                try {
                    best_by_op[op_name] = float(row[0]);
                } except Exception {
                    continue;
                }
            }
        } except Exception as e {
            print(f"Error loading tree best value for {op_name}: {e}");
        }
    }
    return best_by_op;
}

def _has_benchmark_outputs(project_name: str) -> bool {
    bench_path = path.join(
        _project_dir(project_name), "benchmarks", "op_benchmarks.json"
    );
    return path.exists(bench_path);
}

def _has_benchmark_outputs_after(project_name: str, started_at: str = "") -> bool {
    bench_path = path.join(
        _project_dir(project_name), "benchmarks", "op_benchmarks.json"
    );
    return _modified_after(bench_path, started_at);
}

def _read_log_excerpt(log_path: str, max_bytes: int = 131072) -> str {
    if not log_path or not path.exists(log_path) {
        return "";
    }
    try {
        size = path.getsize(log_path);
        with open(log_path, "rb") as f {
            if size > max_bytes {
                f.seek(size - max_bytes);
            }
            data = f.read();
        }
        return data.decode("utf-8", errors="ignore");
    } except Exception {
        return "";
    }
}

def _extract_job_error_summary(log_path: str) -> str {
    log_text = _read_log_excerpt(log_path);
    if not log_text {
        return "";
    }

    lines = log_text.splitlines();
    if len(lines) == 0 {
        return "";
    }

    start_idx = len(lines) - 120;
    if start_idx < 0 {
        start_idx = 0;
    }

    idx = len(lines) - 1;
    while idx >= start_idx {
        line = str(lines[idx]).strip();
        if line {
            line_lower = line.lower();
            if "[system error]" in line_lower
            or "[worker crash]" in line_lower
            or "[compilation failed]" in line_lower
            or "[runtime error" in line_lower
            or "traceback" in line_lower
            or "generation failed" in line_lower
            or "optimization failed" in line_lower
            or "benchmark failed" in line_lower
            or "cuda preflight failed" in line_lower {
                return line;
            }
        }
        idx = idx - 1;
    }

    idx = len(lines) - 1;
    while idx >= start_idx {
        line = str(lines[idx]).strip();
        if line and not line.startswith("[job]") {
            return line;
        }
        idx = idx - 1;
    }
    return "";
}

def _refresh_running_job_state(project_name: str, job_key: str, job: dict) -> bool {
    if not job {
        return False;
    }
    log_path = str(job["log"]) if "log" in job and job["log"] else "";
    log_text = _read_log_excerpt(log_path);
    log_lower = log_text.lower() if log_text else "";

    phase = "";
    message = "";
    progress = None;

    if job_key == "profile" {
        if "src.optimizer.benchmarking.benchmark_ops" in log_text {
            phase = "benchmarking";
            message = "Profiling complete. Running baseline benchmark.";
            progress = 0.8;
        } elif "profiling" in log_lower or "profile" in log_lower {
            phase = "profiling";
            message = "Profiling operators.";
            progress = 0.45;
        } elif "prepar" in log_lower {
            phase = "preparing";
            message = "Preparing profiling assets.";
            progress = 0.15;
        } else {
            phase = "profiling";
            message = "Profiling and baseline benchmarking in progress.";
            progress = 0.2;
        }
    } elif job_key == "generate" {
        if "src.optimizer.benchmarking.benchmark_ops" in log_text {
            phase = "benchmarking";
            message = "Generation complete. Benchmarking kernels.";
            progress = 0.9;
        } elif "src.optimizer.pipeline" in log_text {
            phase = "optimizing";
            message = "Generation complete. Optimizing kernels.";
            progress = 0.65;
        } elif "src.generator.main" in log_text {
            phase = "generating";
            message = "Generating kernels.";
            progress = 0.35;
        } else {
            phase = "generating";
            message = "Queued for kernel generation.";
            progress = 0.1;
        }
    } elif job_key == "optimize" {
        if "src.optimizer.benchmarking.benchmark_ops" in log_text {
            phase = "benchmarking";
            message = "Optimization complete. Benchmarking kernels.";
            progress = 0.9;
        } elif "src.optimizer.pipeline" in log_text {
            phase = "optimizing";
            message = "Optimizing kernels.";
            progress = 0.6;
        } else {
            phase = "optimizing";
            message = "Queued for optimization.";
            progress = 0.1;
        }
    } elif job_key == "benchmark" {
        phase = "benchmarking";
        message = "Benchmarking kernels.";
        progress = 0.5;
    } else {
        return False;
    }

    changed = False;
    if phase and ((not ("phase" in job)) or job["phase"] != phase) {
        job["phase"] = phase;
        changed = True;
    }
    structured_progress = None;
    if (
        "progress" in job
        and job["progress"]
        and "total" in job["progress"]
        and job["progress"]["total"]
        and "percent" in job["progress"]
        and job["progress"]["percent"] is not None
    ) {
        try {
            structured_progress = float(job["progress"]["percent"]);
        } except Exception {
            structured_progress = None;
        }
    }
    if (
        "progress" in job
        and job["progress"]
        and "total" in job["progress"]
        and job["progress"]["total"]
        and "message" in job
        and job["message"]
    ) {
        message = str(job["message"]);
    }
    if message and ((not ("message" in job)) or job["message"] != message) {
        job["message"] = message;
        changed = True;
    }
    if progress is not None {
        current_progress = None;
        if "progress_percent" in job and job["progress_percent"] is not None {
            try {
                current_progress = float(job["progress_percent"]);
            } except Exception {
                current_progress = None;
            }
        }
        target_progress = progress;
        if structured_progress is not None {
            try {
                target_progress = float(structured_progress);
            } except Exception {
                target_progress = progress;
            }
        }
        should_update_progress = False;
        if current_progress is None {
            should_update_progress = True;
        } elif structured_progress is not None {
            try {
                delta = float(target_progress) - float(current_progress);
                should_update_progress = delta > 0.000001 or delta < -0.000001;
            } except Exception {
                should_update_progress = False;
            }
        } elif target_progress > current_progress {
            should_update_progress = True;
        }
        if should_update_progress {
            job["progress_percent"] = target_progress;
            changed = True;
        }
    }
    if changed {
        job["updated_at"] = datetime.now().isoformat();
    }
    return bool(changed);
}

def _start_job(project_name: str, job_key: str, cmd: list, log_name: str) -> dict {
    state = _read_state(project_name);
    current = state[job_key] if state and job_key in state else {};
    current_status = (
        str(current["status"]).strip().lower()
        if current and "status" in current and current["status"]
        else ""
    );
    current_pid = current["pid"] if current and "pid" in current else None;

    if (
        current_status == "running"
        or current_status == "queued"
        or current_status == "paused"
    )
    and _pid_is_alive(current_pid) {
        return {
            "success": True,
            "skipped": True,
            "reason": "in_progress",
            "message": f"{job_key} job already in progress.",
            "pid": current_pid
        };
    }

    if job_key == "profile" and _has_profile_outputs(project_name) {
        updates = {
            "status": "completed",
            "control": "idle",
            "active": False,
            "phase": "done",
            "progress_percent": 1.0,
            "message": "Profile and baseline benchmark already available."
        };
        if current and "run_id" in current and current["run_id"] {
            updates["run_id"] = current["run_id"];
        }
        _update_job_state(project_name, "profile", updates);
        return {
            "success": True,
            "skipped": True,
            "reason": "already_profiled",
            "message": "Profile and baseline benchmark already available."
        };
    }

    project_dir = _project_dir(project_name);
    logs_dir = path.join(project_dir, "logs");
    _ensure_dir(logs_dir);
    log_path = path.join(logs_dir, log_name);
    run_id = f"{job_key}:{int(datetime.now().timestamp() * 1000)}";
    initial_phase = "running";
    initial_message = "Job started";
    initial_progress = 0.0;
    if job_key == "profile" {
        initial_phase = "profiling";
        initial_message = "Profiling and baseline benchmarking started.";
        initial_progress = 0.1;
    } elif job_key == "generate" {
        initial_phase = "generating";
        initial_message = "Kernel generation started.";
        initial_progress = 0.05;
    } elif job_key == "optimize" {
        initial_phase = "optimizing";
        initial_message = "Kernel optimization started.";
        initial_progress = 0.05;
    } elif job_key == "benchmark" {
        initial_phase = "benchmarking";
        initial_message = "Kernel benchmark started.";
        initial_progress = 0.05;
    }

    _update_job_state(
        project_name,
        job_key,
        {
            "status": "running",
            "control": "running",
            "active": True,
            "phase": initial_phase,
            "message": initial_message,
            "progress_percent": initial_progress,
            "started_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "finished_at": None,
            "log": log_path,
            "command": cmd,
            "run_id": run_id
        }
    );

    with open(log_path, "w") as f {
        f.write(f"[job] Started at {datetime.now().isoformat()}\n");
        f.write(f"[job] Command: {cmd}\n");
    }
    child_env = os.environ.copy();
    child_env["CGINS_STATE_PATH"] = path.join(_project_dir(project_name), "state.json");
    child_env["CGINS_JOB_KEY"] = job_key;
    log_file = open(log_path, "a");
    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=child_env
    );
    log_file.close();
    _update_job_state(project_name, job_key, {"pid": proc.pid});
    return {"success": True, "pid": proc.pid, "log": log_path};
}

def _start_job_container(
    project_name: str, job_key: str, cmd: list, log_name: str, image: str
) -> dict {
    return _start_job(project_name, job_key, cmd, log_name);
}

def _find_optimized_dir(project_name: str) -> str {
    project_trees = path.join(_project_dir(project_name), "trees");
    if path.exists(project_trees) and path.isdir(project_trees) {
        for name in listdir(project_trees) {
            full_path = path.join(project_trees, name);
            if not path.isdir(full_path) {
                continue;
            }
            if path.exists(path.join(full_path, "nodes.db")) {
                return project_trees;
            }
            kernels_dir = path.join(full_path, "kernels");
            if path.exists(kernels_dir) and path.isdir(kernels_dir) {
                for fname in listdir(kernels_dir) {
                    if fname.endswith(".cu")
                    or fname.endswith(".py")
                    or fname.endswith(".metal") {
                        return project_trees;
                    }
                }
            }
            if path.exists(path.join(full_path, "generated_root.json")) {
                return project_trees;
            }
        }
    }

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

# --- Canonical Metadata DB (project + metrics) ---
def _catalog_db_path() -> str {
    return path.join(_projects_base_dir(), "catalog.db");
}

def _ensure_catalog_schema() -> None {
    db_path = _catalog_db_path();
    conn = sqlite3.connect(db_path);
    cursor = conn.cursor();
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            name TEXT PRIMARY KEY,
            project_path TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    );
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS operator_metrics (
            project_name TEXT NOT NULL,
            op_name TEXT NOT NULL,
            op_count INTEGER NOT NULL DEFAULT 0,
            pytorch_ms REAL,
            forged_ms REAL,
            winner TEXT,
            total_time_ms REAL NOT NULL DEFAULT 0,
            updated_at TEXT,
            PRIMARY KEY (project_name, op_name)
        )
        """
    );
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS project_metric_meta (
            project_name TEXT PRIMARY KEY,
            status TEXT,
            profile_status TEXT,
            profile_message TEXT,
            errors_json TEXT,
            updated_at TEXT
        )
        """
    );
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS run_state (
            project_name TEXT NOT NULL,
            job_key TEXT NOT NULL,
            status TEXT,
            control TEXT,
            active INTEGER NOT NULL DEFAULT 0,
            phase TEXT,
            message TEXT,
            progress_percent REAL,
            started_at TEXT,
            updated_at TEXT,
            finished_at TEXT,
            log TEXT,
            command_json TEXT,
            run_id TEXT,
            pid INTEGER,
            progress_json TEXT,
            extras_json TEXT,
            PRIMARY KEY (project_name, job_key)
        )
        """
    );
    conn.commit();
    conn.close();
}

def _upsert_project_catalog(project_name: str, project_path: str = "") -> None {
    if not project_name {
        return;
    }
    _ensure_catalog_schema();
    now = datetime.now().isoformat();
    db_path = _catalog_db_path();
    conn = sqlite3.connect(db_path);
    cursor = conn.cursor();
    created_at = now;
    cursor.execute(
        "SELECT created_at FROM projects WHERE name = ?",
        (project_name,)
    );
    existing = cursor.fetchone();
    if existing and len(existing) > 0 and existing[0] {
        created_at = str(existing[0]);
    }
    cursor.execute(
        """
        INSERT OR REPLACE INTO projects(name, project_path, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        (project_name, project_path, created_at, now)
    );
    conn.commit();
    conn.close();
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

def _load_project_catalog() -> list {
    _ensure_catalog_schema();
    db_path = _catalog_db_path();
    conn = sqlite3.connect(db_path);
    cursor = conn.cursor();
    cursor.execute(
        "SELECT name, project_path, created_at, updated_at FROM projects ORDER BY lower(name)"
    );
    rows = cursor.fetchall();
    conn.close();
    out = [];
    for row in rows {
        if not row or len(row) < 1 or not row[0] {
            continue;
        }
        out.append(
            {
                "name": str(row[0]),
                "project_path": str(row[1]) if len(row) > 1 and row[1] else "",
                "created_at": str(row[2]) if len(row) > 2 and row[2] else "",
                "updated_at": str(row[3]) if len(row) > 3 and row[3] else ""
            }
        );
    }
    return out;
}

def _json_load_safe(raw: str, default_value: Any) -> Any {
    if not raw {
        return default_value;
    }
    try {
        return loads(raw);
    } except Exception {
        return default_value;
    }
}

def _write_job_state_db(project_name: str, job_key: str, job_state: dict) -> None {
    if not project_name or not job_key or not job_state {
        return;
    }
    _ensure_catalog_schema();
    now = datetime.now().isoformat();
    status = str(job_state["status"]) if "status" in job_state and job_state["status"] else "";
    control = str(job_state["control"]) if "control" in job_state and job_state["control"] else "";
    active = 1 if ("active" in job_state and job_state["active"]) else 0;
    phase = str(job_state["phase"]) if "phase" in job_state and job_state["phase"] else "";
    message = str(job_state["message"]) if "message" in job_state and job_state["message"] else "";
    progress_percent = None;
    if "progress_percent" in job_state and job_state["progress_percent"] is not None {
        try {
            progress_percent = float(job_state["progress_percent"]);
        } except Exception {
            progress_percent = None;
        }
    }
    started_at = str(job_state["started_at"]) if "started_at" in job_state and job_state["started_at"] else "";
    updated_at = str(job_state["updated_at"]) if "updated_at" in job_state and job_state["updated_at"] else now;
    finished_at = str(job_state["finished_at"]) if "finished_at" in job_state and job_state["finished_at"] else "";
    log_path = str(job_state["log"]) if "log" in job_state and job_state["log"] else "";
    run_id = str(job_state["run_id"]) if "run_id" in job_state and job_state["run_id"] else "";
    pid_val = None;
    if "pid" in job_state and job_state["pid"] is not None and str(job_state["pid"]) != "" {
        try {
            pid_val = int(job_state["pid"]);
        } except Exception {
            pid_val = None;
        }
    }
    command_json = "[]";
    if "command" in job_state and job_state["command"] is not None {
        try {
            command_json = dumps(job_state["command"]);
        } except Exception {
            command_json = "[]";
        }
    }
    progress_json = "{}";
    if "progress" in job_state and job_state["progress"] is not None {
        try {
            progress_json = dumps(job_state["progress"]);
        } except Exception {
            progress_json = "{}";
        }
    }
    known_keys = [
        "status",
        "control",
        "active",
        "phase",
        "message",
        "progress_percent",
        "started_at",
        "updated_at",
        "finished_at",
        "log",
        "command",
        "run_id",
        "pid",
        "progress"
    ];
    extras = {};
    for key in job_state {
        if key in known_keys {
            continue;
        }
        extras[key] = job_state[key];
    }
    extras_json = "{}";
    try {
        extras_json = dumps(extras);
    } except Exception {
        extras_json = "{}";
    }

    db_path = _catalog_db_path();
    conn = sqlite3.connect(db_path);
    cursor = conn.cursor();
    cursor.execute(
        """
        INSERT OR REPLACE INTO run_state(
            project_name, job_key, status, control, active, phase, message,
            progress_percent, started_at, updated_at, finished_at, log,
            command_json, run_id, pid, progress_json, extras_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            project_name,
            job_key,
            status,
            control,
            active,
            phase,
            message,
            progress_percent,
            started_at,
            updated_at,
            finished_at,
            log_path,
            command_json,
            run_id,
            pid_val,
            progress_json,
            extras_json
        )
    );
    conn.commit();
    conn.close();
}

def _write_project_run_state_db(project_name: str, state: dict) -> None {
    if not project_name or not state or len(state) == 0 {
        return;
    }
    _ensure_catalog_schema();
    db_path = _catalog_db_path();
    conn = sqlite3.connect(db_path);
    cursor = conn.cursor();
    cursor.execute("DELETE FROM run_state WHERE project_name = ?", (project_name,));
    conn.commit();
    conn.close();
    for job_key in state {
        job_state = state[job_key];
        if not job_state or not ("status" in job_state) {
            continue;
        }
        _write_job_state_db(project_name, str(job_key), job_state);
    }
}

def _read_project_run_state_db(project_name: str) -> dict {
    if not project_name {
        return {};
    }
    _ensure_catalog_schema();
    db_path = _catalog_db_path();
    conn = sqlite3.connect(db_path);
    cursor = conn.cursor();
    cursor.execute(
        """
        SELECT
            job_key, status, control, active, phase, message, progress_percent,
            started_at, updated_at, finished_at, log, command_json, run_id, pid,
            progress_json, extras_json
        FROM run_state
        WHERE project_name = ?
        ORDER BY job_key
        """,
        (project_name,)
    );
    rows = cursor.fetchall();
    conn.close();

    state = {};
    for row in rows {
        if not row or len(row) < 16 or not row[0] {
            continue;
        }
        job_key = str(row[0]);
        job = {};
        job["status"] = str(row[1]) if row[1] else "";
        job["control"] = str(row[2]) if row[2] else "";
        job["active"] = bool(row[3]) if row[3] is not None else False;
        job["phase"] = str(row[4]) if row[4] else "";
        job["message"] = str(row[5]) if row[5] else "";
        if row[6] is not None {
            try {
                job["progress_percent"] = float(row[6]);
            } except Exception as e {
                _ = str(e);
            }
        }
        if row[7] {
            job["started_at"] = str(row[7]);
        }
        if row[8] {
            job["updated_at"] = str(row[8]);
        }
        if row[9] {
            job["finished_at"] = str(row[9]);
        }
        if row[10] {
            job["log"] = str(row[10]);
        }
        if row[11] {
            job["command"] = _json_load_safe(str(row[11]), []);
        }
        if row[12] {
            job["run_id"] = str(row[12]);
        }
        if row[13] is not None {
            try {
                job["pid"] = int(row[13]);
            } except Exception as e {
                _ = str(e);
            }
        }
        if row[14] {
            job["progress"] = _json_load_safe(str(row[14]), {});
        }
        extras = _json_load_safe(str(row[15]) if row[15] else "{}", {});
        if extras and type(extras) == dict {
            for extra_key in extras {
                if not (extra_key in job) {
                    job[extra_key] = extras[extra_key];
                }
            }
        }
        state[job_key] = job;
    }
    return state;
}

def _upsert_project_metric_rows(project_name: str, payload: dict) -> None {
    if not project_name {
        return;
    }
    _ensure_catalog_schema();
    now = datetime.now().isoformat();
    rows = payload["rows"] if payload and "rows" in payload and payload["rows"] else [];
    status = str(payload["status"]) if payload and "status" in payload and payload["status"] else "empty";
    profile_status = (
        str(payload["profile_status"])
        if payload and "profile_status" in payload and payload["profile_status"]
        else ""
    );
    profile_message = (
        str(payload["profile_message"])
        if payload and "profile_message" in payload and payload["profile_message"]
        else ""
    );
    errors_json = "[]";
    if payload and "errors" in payload and payload["errors"] {
        try {
            errors_json = dumps(payload["errors"]);
        } except Exception {
            errors_json = "[]";
        }
    }

    db_path = _catalog_db_path();
    conn = sqlite3.connect(db_path);
    cursor = conn.cursor();

    cursor.execute(
        "DELETE FROM operator_metrics WHERE project_name = ?",
        (project_name,)
    );
    for row in rows {
        if not row {
            continue;
        }
        op_name = str(row["name"]) if "name" in row and row["name"] else "";
        if not op_name {
            continue;
        }
        op_count = 0;
        try {
            op_count = int(row["count"]) if "count" in row and row["count"] else 0;
        } except Exception {
            op_count = 0;
        }
        pytorch_ms = None;
        if "pytorch_ms" in row and row["pytorch_ms"] is not None {
            try {
                pytorch_ms = float(row["pytorch_ms"]);
            } except Exception {
                pytorch_ms = None;
            }
        }
        forged_ms = None;
        if "forged_ms" in row and row["forged_ms"] is not None {
            try {
                forged_ms = float(row["forged_ms"]);
            } except Exception {
                forged_ms = None;
            }
        }
        total_time_ms = 0.0;
        try {
            total_time_ms = (
                float(row["total_time_ms"])
                if "total_time_ms" in row and row["total_time_ms"] is not None
                else 0.0
            );
        } except Exception {
            total_time_ms = 0.0;
        }
        winner = str(row["winner"]) if "winner" in row and row["winner"] else "";
        cursor.execute(
            """
            INSERT OR REPLACE INTO operator_metrics(
                project_name, op_name, op_count, pytorch_ms, forged_ms, winner,
                total_time_ms, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_name,
                op_name,
                op_count,
                pytorch_ms,
                forged_ms,
                winner,
                total_time_ms,
                now
            )
        );
    }

    cursor.execute(
        """
        INSERT OR REPLACE INTO project_metric_meta(
            project_name, status, profile_status, profile_message, errors_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (project_name, status, profile_status, profile_message, errors_json, now)
    );
    conn.commit();
    conn.close();
}

def _read_project_metric_rows(project_name: str) -> dict {
    _ensure_catalog_schema();
    db_path = _catalog_db_path();
    conn = sqlite3.connect(db_path);
    cursor = conn.cursor();
    cursor.execute(
        """
        SELECT op_name, op_count, pytorch_ms, forged_ms, winner, total_time_ms
        FROM operator_metrics
        WHERE project_name = ?
        ORDER BY op_name
        """,
        (project_name,)
    );
    metric_rows = cursor.fetchall();

    cursor.execute(
        """
        SELECT status, profile_status, profile_message, errors_json
        FROM project_metric_meta
        WHERE project_name = ?
        """,
        (project_name,)
    );
    meta = cursor.fetchone();
    conn.close();

    rows = [];
    rows_by_op = {};
    speed_comparison = [];
    operator_usage = [];

    for m in metric_rows {
        if not m or len(m) < 6 or not m[0] {
            continue;
        }
        op_name = str(m[0]);
        count = 0;
        try {
            count = int(m[1]) if m[1] is not None else 0;
        } except Exception {
            count = 0;
        }
        pytorch_ms = None;
        if m[2] is not None {
            try {
                parsed = float(m[2]);
                if parsed > 0.0 {
                    pytorch_ms = parsed;
                }
            } except Exception {
                pytorch_ms = None;
            }
        }
        forged_ms = None;
        if m[3] is not None {
            try {
                parsed = float(m[3]);
                if parsed > 0.0 {
                    forged_ms = parsed;
                }
            } except Exception {
                forged_ms = None;
            }
        }
        winner = str(m[4]) if m[4] else "unknown";
        total_time_ms = 0.0;
        try {
            total_time_ms = float(m[5]) if m[5] is not None else 0.0;
        } except Exception {
            total_time_ms = 0.0;
        }

        row = {
            "name": op_name,
            "count": count,
            "winner": winner,
            "total_time_ms": total_time_ms
        };
        if pytorch_ms is not None {
            row["pytorch_ms"] = pytorch_ms;
        }
        if forged_ms is not None {
            row["forged_ms"] = forged_ms;
        }

        rows.append(row);
        rows_by_op[op_name] = row;

        if pytorch_ms is not None {
            srow = {"name": op_name, "pytorch": pytorch_ms};
            if forged_ms is not None {
                srow["forged"] = forged_ms;
                srow["optimized"] = forged_ms;
            }
            speed_comparison.append(srow);
        }
        if total_time_ms > 0.0 {
            operator_usage.append(
                {"name": op_name, "count": count, "total_time_ms": total_time_ms}
            );
        }
    }

    status = "empty";
    profile_status = "";
    profile_message = "";
    errors = [];
    meta_vals = [];
    if meta is not None {
        try {
            meta_vals = list(meta);
        } except Exception {
            meta_vals = [];
        }
    }
    if len(meta_vals) > 0 and meta_vals[0] {
        status = str(meta_vals[0]);
    }
    if len(meta_vals) > 1 and meta_vals[1] {
        profile_status = str(meta_vals[1]);
    }
    if len(meta_vals) > 2 and meta_vals[2] {
        profile_message = str(meta_vals[2]);
    }
    if len(meta_vals) > 3 and meta_vals[3] {
        try {
            parsed_errors = loads(str(meta_vals[3]));
            if parsed_errors {
                errors = parsed_errors;
            }
        } except Exception {
            errors = [];
        }
    }

    return {
        "rows": rows,
        "rows_by_op": rows_by_op,
        "operator_usage": operator_usage,
        "speed_comparison": speed_comparison,
        "status": status,
        "profile_status": profile_status,
        "profile_message": profile_message,
        "errors": errors
    };
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
            _migrate_legacy_projects();
            projects_dir = _projects_base_dir();
            _ensure_dir(projects_dir);

            project_path = path.join(projects_dir, self.projectName);
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
            "-m",
            "src.optimizer.workflow",
            "profile",
            "--project",
            self.projectName
        ];
        result = _start_job(self.projectName, "profile", cmd, "profile.log");
        report result;
    }
}

walker:pub StartGenerate {
    has projectName: str,
        ops: list = [],
        benchmark: bool = False,
        iterations: int = 0,
        targetDevice: str = "cuda";

    can run with entry {
        io_dir = path.join(_project_dir(self.projectName), "io", "individual_ops");
        out_dir = path.join(_project_dir(self.projectName), "kernels", "generated");
        _ensure_dir(out_dir);
        target_device = str(self.targetDevice).lower() if self.targetDevice else "cuda";
        if not self.targetDevice {
            cfg_path = path.join(_project_dir(self.projectName), "config.json");
            if path.exists(cfg_path) {
                try {
                    with open(cfg_path, "r") as f {
                        project_cfg = load(f);
                    }
                    if project_cfg
                    and "generator" in project_cfg
                    and "target_device" in project_cfg["generator"] {
                        target_device = str(project_cfg["generator"]["target_device"]).lower();
                    }
                } except Exception as e {
                    print(f"Failed reading config target_device: {e}");
                }
            }
        }
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
        ops_list = [];
        for o in self.ops {
            if o {
                ops_list.append(str(o));
            }
        }
        if len(ops_list) == 0 and path.exists(io_dir) {
            for op_name in listdir(io_dir) {
                op_dir = path.join(io_dir, op_name);
                if path.isdir(op_dir) {
                    ops_list.append(str(op_name));
                }
            }
        }
        cmd = [
            sys.executable,
            "-m",
            "src.optimizer.workflow",
            "generate",
            "--project",
            self.projectName,
            "--ops",
            ",".join(ops_list),
            "--target-device",
            target_device
        ];
        if self.benchmark {
            cmd.append("--benchmark");
        }
        if self.iterations > 0 {
            cmd.append("--iterations");
            cmd.append(str(self.iterations));
        }
        result = _start_job(self.projectName, "generate", cmd, "generate.log");
        report result;
    }
}

walker:pub StartOptimize {
    has projectName: str,
        ops: list = [],
        iterations: int = 3,
        benchmark: bool = False;

    can run with entry {
        io_dir = path.join(_project_dir(self.projectName), "io", "individual_ops");
        ops_list = [];
        for o in self.ops {
            if o {
                ops_list.append(str(o));
            }
        }
        if len(ops_list) == 0 and path.exists(io_dir) {
            for op_name in listdir(io_dir) {
                op_dir = path.join(io_dir, op_name);
                if path.isdir(op_dir) {
                    ops_list.append(str(op_name));
                }
            }
        }
        target_device = "cuda";
        cfg_path = path.join(_project_dir(self.projectName), "config.json");
        if path.exists(cfg_path) {
            try {
                with open(cfg_path, "r") as f {
                    project_cfg = load(f);
                }
                if project_cfg
                and "generator" in project_cfg
                and "target_device" in project_cfg["generator"] {
                    target_device = str(project_cfg["generator"]["target_device"]).lower();
                }
            } except Exception as e {
                print(f"Failed reading config target_device: {e}");
            }
        }
        cmd = [
            sys.executable,
            "-m",
            "src.optimizer.workflow",
            "optimize",
            "--project",
            self.projectName,
            "--ops",
            ",".join(ops_list),
            "--target-device",
            target_device
        ];
        if self.iterations > 0 {
            cmd.append("--iterations");
            cmd.append(str(self.iterations));
        }
        if self.benchmark {
            cmd.append("--benchmark");
        }
        result = _start_job(self.projectName, "optimize", cmd, "optimize.log");
        report result;
    }
}

walker:pub StartBenchmark {
    has projectName: str;

    can run with entry {
        cmd = [
            sys.executable,
            "-m",
            "src.optimizer.benchmarking.benchmark_ops",
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
        opt_kernel_dir = path.join(opt_dir, "kernels") if opt_dir else "";
        opt_kernels = (
            _list_files(opt_kernel_dir, [".cu", ".py", ".metal", ".so"])
            if opt_kernel_dir and path.exists(opt_kernel_dir)
            else _list_files(opt_dir, [".cu", ".py", ".metal", ".so"])
        );
        opt_logs = _list_files(opt_dir, [".txt", ".log", ".json"]);

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
        valid_ids = {};

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
                node_val = None;
                try {
                    node_val = float(row[2]) if row[2] is not None else None;
                } except Exception {
                    node_val = None;
                }
                if node_val is None or node_val <= 0.0 {
                    continue;
                }
                node_code = row[6];
                code_text = str(node_code).strip() if node_code is not None else "";
                if code_text and "\n" not in code_text and (
                    code_text.endswith(".cu")
                    or code_text.endswith(".py")
                    or code_text.endswith(".metal")
                    or "/kernels/" in code_text
                ) {
                    resolved_inline_code = False;
                    path_candidates = [];
                    if path.isabs(code_text) {
                        path_candidates.append(code_text);
                    } else {
                        path_candidates.append(path.join(project_dir, "trees", code_text));
                        path_candidates.append(path.join(project_dir, code_text));
                        path_candidates.append(path.join(repo_root, code_text));
                        path_candidates.append(
                            path.join(
                                repo_root, "kernels", "projects", self.projectName, code_text
                            )
                        );
                        path_candidates.append(
                            path.join(
                                repo_root,
                                "kernels",
                                "projects",
                                self.projectName,
                                "trees",
                                code_text
                            )
                        );
                        path_candidates.append(
                            path.join(
                                project_dir, "kernels", "generated", "individual_op_kernels", code_text
                            )
                        );
                        path_candidates.append(
                            path.join(
                                project_dir,
                                "kernels",
                                "generated",
                                "individual_op_kernels",
                                self.opName,
                                code_text
                            )
                        );
                        path_candidates.append(
                            path.join(
                                project_dir,
                                "kernels",
                                "generated",
                                "individual_op_kernels",
                                self.opName,
                                "kernel",
                                code_text
                            )
                        );
                        path_candidates.append(
                            path.join(
                                project_dir,
                                "kernels",
                                "generated",
                                "individual_op_kernels",
                                self.opName,
                                "kernels",
                                code_text
                            )
                        );
                    }
                    for p in path_candidates {
                        if not path.exists(p) {
                            continue;
                        }
                        try {
                            with open(p, "r", encoding="utf-8", errors="ignore") as f {
                                node_code = f.read();
                            }
                            resolved_inline_code = True;
                            break;
                        } except Exception {
                            continue;
                        }
                    }
                    if not resolved_inline_code {
                        node_code = None;
                    }
                }
                if node_code is None or str(node_code).strip() == "" {
                    kernel_dirs = [
                        path.join(project_dir, "trees", self.opName, "kernels"),
                        path.join(
                            project_dir,
                            "kernels",
                            "generated",
                            "individual_op_kernels",
                            self.opName,
                            "kernel"
                        ),
                        path.join(
                            project_dir,
                            "kernels",
                            "generated",
                            "individual_op_kernels",
                            self.opName,
                            "kernels"
                        ),
                        path.join(
                            project_dir,
                            "kernels",
                            "generated",
                            "individual_op_kernels",
                            self.opName
                        )
                    ];
                    candidates = [
                        f"kernel_{row[0]}.cu",
                        f"kernel_{row[0]}.py",
                        f"kernel_{row[0]}.metal",
                        f"kernel-{row[0]}-g.cu",
                        "kernel_0.cu",
                        "kernel_0.py",
                        "kernel_0.metal",
                        "kernel-0-g.cu",
                        "kernel.cu",
                        "cuda.cu"
                    ];
                    for kernels_dir in kernel_dirs {
                        if not path.exists(kernels_dir) {
                            continue;
                        }
                        for fname in candidates {
                            candidate_path = path.join(kernels_dir, fname);
                            if not path.exists(candidate_path) {
                                continue;
                            }
                            try {
                                with open(
                                    candidate_path, "r", encoding="utf-8", errors="ignore"
                                ) as f {
                                    node_code = f.read();
                                }
                                break;
                            } except Exception {
                                continue;
                            }
                        }
                        if node_code is not None and str(node_code).strip() != "" {
                            break;
                        }
                    }
                }
                valid_ids[row[0]] = True;
                mcts_nodes.append(
                    {
                        "id": row[0],
                        "visits": row[1],
                        "value": node_val,
                        "best_subtree_value": row[3],
                        "improvement_description": row[4],
                        "timestamp": row[5],
                        "code": node_code
                    }
                );
            }

            # Fetch edges
            cursor.execute("SELECT parent_id, child_id FROM edges");
            rows = cursor.fetchall();
            for row in rows {
                if not (row[0] in valid_ids) or not (row[1] in valid_ids) {
                    continue;
                }
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

                        cursor.execute("SELECT COUNT(*) FROM nodes WHERE value IS NOT NULL AND value > 0");
                        node_count = cursor.fetchone()[0];

                        cursor.execute("SELECT MIN(value) FROM nodes WHERE value IS NOT NULL AND value > 0");
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
        payload = _collect_project_metrics(self.projectName);
        report {
            "operator_usage": (
                payload["operator_usage"] if payload and "operator_usage" in payload else []
            ),
            "speed_comparison": (
                payload["speed_comparison"]
                if payload and "speed_comparison" in payload
                else []
            ),
            "status": payload["status"] if payload and "status" in payload else "empty",
            "profile_status": (
                payload["profile_status"] if payload and "profile_status" in payload else ""
            ),
            "profile_message": (
                payload["profile_message"]
                if payload and "profile_message" in payload
                else ""
            ),
            "errors": payload["errors"] if payload and "errors" in payload else []
        };
    }
}

def _normalize_metric_op_name(raw: Any) -> str {
    text = str(raw).strip().lower() if raw is not None else "";
    if not text {
        return "";
    }
    prefix = "torch.nn.functional.";
    if text.startswith(prefix) {
        text = "torch_nn_functional_" + text.replace(prefix, "");
    }
    return text.replace(".", "_").replace("/", "_");
}

def _to_positive_float(raw: Any) -> Any {
    if raw is None {
        return None;
    }
    try {
        val = float(raw);
        if val > 0.0 {
            return val;
        }
    } except Exception {
        _ = 0;
    }
    return None;
}

def _to_int(raw: Any) -> int {
    try {
        return int(raw);
    } except Exception {
        return 0;
    }
}

def _collect_project_metrics(project_name: str) -> dict {
    project_dir = _project_dir(project_name);

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

    count_by_op = {};
    for full_name in op_counts {
        op_name = _normalize_metric_op_name(full_name);
        if not op_name {
            continue;
        }
        count_by_op[op_name] = _to_int(op_counts[full_name]);
    }

    profile_ms_by_op = {};
    for full_name in op_profile_ms {
        op_name = _normalize_metric_op_name(full_name);
        if not op_name {
            continue;
        }
        ms_val = _to_positive_float(op_profile_ms[full_name]);
        if ms_val is not None {
            profile_ms_by_op[op_name] = ms_val;
        }
    }

    bench_by_op = {};
    bench_order = [];
    seen_bench = {};
    for r in bench_results {
        if not r {
            continue;
        }
        op_name = "";
        if "op" in r and r["op"] {
            op_name = _normalize_metric_op_name(r["op"]);
        } elif "name" in r and r["name"] {
            op_name = _normalize_metric_op_name(r["name"]);
        }
        if not op_name {
            continue;
        }
        bench_by_op[op_name] = r;
        if not (op_name in seen_bench) {
            bench_order.append(op_name);
            seen_bench[op_name] = True;
        }
    }

    tree_best_by_op = {};
    tree_raw = _load_tree_best_by_op(project_name);
    for op_name in tree_raw {
        norm_name = _normalize_metric_op_name(op_name);
        if not norm_name {
            continue;
        }
        ms_val = _to_positive_float(tree_raw[op_name]);
        if ms_val is None {
            continue;
        }
        if not (norm_name in tree_best_by_op)
        or ms_val < tree_best_by_op[norm_name] {
            tree_best_by_op[norm_name] = ms_val;
        }
    }

    baseline_by_op = {};
    for op_name in profile_ms_by_op {
        baseline_by_op[op_name] = profile_ms_by_op[op_name];
    }
    for op_name in bench_by_op {
        b_row = bench_by_op[op_name];
        pytorch_raw = None;
        if "pytorch_ms" in b_row and b_row["pytorch_ms"] is not None {
            pytorch_raw = b_row["pytorch_ms"];
        } elif "pytorch" in b_row and b_row["pytorch"] is not None {
            pytorch_raw = b_row["pytorch"];
        } elif "baseline" in b_row and b_row["baseline"] is not None {
            pytorch_raw = b_row["baseline"];
        }
        ms_val = _to_positive_float(pytorch_raw);
        if ms_val is not None {
            baseline_by_op[op_name] = ms_val;
        }
    }

    forged_best_by_op = {};
    for op_name in bench_by_op {
        b_row = bench_by_op[op_name];
        kernel_status = (
            str(b_row["kernel_status"]).lower()
            if "kernel_status" in b_row and b_row["kernel_status"]
            else ""
        );
        if kernel_status != "ok" {
            continue;
        }
        kernel_ms = (
            _to_positive_float(b_row["kernel_ms"])
            if "kernel_ms" in b_row
            else None
        );
        if kernel_ms is not None {
            forged_best_by_op[op_name] = kernel_ms;
        }
    }
    for op_name in tree_best_by_op {
        tree_ms = tree_best_by_op[op_name];
        if not (op_name in forged_best_by_op) {
            forged_best_by_op[op_name] = tree_ms;
        } elif tree_ms < forged_best_by_op[op_name] {
            forged_best_by_op[op_name] = tree_ms;
        }
    }

    all_ops = [];
    seen_ops = {};
    for op_name in bench_order {
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
    for op_name in baseline_by_op {
        if not (op_name in seen_ops) {
            all_ops.append(op_name);
            seen_ops[op_name] = True;
        }
    }
    for op_name in forged_best_by_op {
        if not (op_name in seen_ops) {
            all_ops.append(op_name);
            seen_ops[op_name] = True;
        }
    }

    rows = [];
    rows_by_op = {};
    speed_comparison = [];
    operator_usage = [];

    for op_name in all_ops {
        b_row = bench_by_op[op_name] if op_name in bench_by_op else {};
        count = count_by_op[op_name] if op_name in count_by_op else 0;
        if count <= 0 and b_row and "entries" in b_row and b_row["entries"] {
            count = _to_int(b_row["entries"]);
        }
        if count < 0 {
            count = 0;
        }

        pytorch_ms = baseline_by_op[op_name] if op_name in baseline_by_op else None;
        forged_ms = (
            forged_best_by_op[op_name] if op_name in forged_best_by_op else None
        );

        winner = "unknown";
        if pytorch_ms is not None {
            winner = "pytorch";
        }
        if forged_ms is not None and (pytorch_ms is None or forged_ms < pytorch_ms) {
            winner = "forged";
        }

        effective_ms = forged_ms if forged_ms is not None else pytorch_ms;
        total_time_ms = effective_ms * count if effective_ms is not None and count > 0 else 0.0;

        row = {
            "name": op_name,
            "count": count,
            "winner": winner,
            "total_time_ms": total_time_ms
        };
        if pytorch_ms is not None {
            row["pytorch_ms"] = pytorch_ms;
        }
        if forged_ms is not None {
            row["forged_ms"] = forged_ms;
        }
        rows.append(row);
        rows_by_op[op_name] = row;

        if pytorch_ms is not None {
            speed_row = {"name": op_name, "pytorch": pytorch_ms};
            if forged_ms is not None {
                speed_row["forged"] = forged_ms;
                speed_row["optimized"] = forged_ms;
            }
            speed_comparison.append(speed_row);
        }

        if total_time_ms > 0.0 {
            operator_usage.append(
                {"name": op_name, "count": count, "total_time_ms": total_time_ms}
            );
        }
    }

    state = _read_state(project_name);
    profile_status = "";
    profile_message = "";
    if state and "profile" in state and state["profile"] {
        p = state["profile"];
        profile_status = str(p["status"]) if "status" in p and p["status"] else "";
        profile_message = (
            str(p["message"]) if "message" in p and p["message"] else ""
        );
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

    computed_payload = {
        "rows": rows,
        "rows_by_op": rows_by_op,
        "operator_usage": operator_usage,
        "speed_comparison": speed_comparison,
        "status": chart_status,
        "profile_status": profile_status,
        "profile_message": profile_message,
        "errors": bench_errors
    };
    _upsert_project_catalog(project_name, project_dir);
    _upsert_project_metric_rows(project_name, computed_payload);
    persisted = _read_project_metric_rows(project_name);
    if persisted and "status" in persisted and persisted["status"] {
        return persisted;
    }
    return computed_payload;
}

walker:pub GetProjectMetrics {
    has projectName: str;

    can run with entry {
        payload = _collect_project_metrics(self.projectName);
        report payload;
    }
}

walker:pub GetOperatorMetrics {
    has projectName: str,
        opName: str;

    can run with entry {
        payload = _collect_project_metrics(self.projectName);
        op_key = _normalize_metric_op_name(self.opName);
        rows_by_op = payload["rows_by_op"] if payload and "rows_by_op" in payload else {};
        metric = rows_by_op[op_key] if op_key in rows_by_op else None;
        report {
            "success": bool(metric),
            "op": op_key,
            "metric": metric,
            "status": payload["status"] if payload and "status" in payload else "empty",
            "profile_status": (
                payload["profile_status"] if payload and "profile_status" in payload else ""
            ),
            "profile_message": (
                payload["profile_message"]
                if payload and "profile_message" in payload
                else ""
            ),
            "errors": payload["errors"] if payload and "errors" in payload else []
        };
    }
}
