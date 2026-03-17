"""Job/process lifecycle helpers and metrics computation."""

import from json { load, loads, dump, dumps }
import from os { path, makedirs, listdir }
import from datetime { datetime }
import os;
import sys;
import subprocess;
import shutil;
import sqlite3;

import from .globals { repo_root, config_store }
import from .catalog_db {
    _ensure_dir,
    _projects_base_dir,
    _json_load_safe,
    _upsert_project_catalog,
    _write_job_state_db,
    _write_project_run_state_db,
    _read_project_run_state_db,
    _upsert_project_metric_rows,
    _read_project_metric_rows
}

def:pub _project_dir(name: str) -> str {
    _migrate_legacy_projects();
    return path.join(_projects_base_dir(), name);
}

def:pub _migrate_legacy_projects() -> None {
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

def:pub _all_projects_dirs() -> list {
    _migrate_legacy_projects();
    return [_projects_base_dir()];
}

def:pub _detect_hardware -> dict {
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

def:pub _apply_llm_env_from_config(
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

def:pub _normalize_job_flags(project_name: str, state: dict, state_path: str) -> dict {
    if not state {
        return {};
    }
    changed = False;
    refresh_metrics = False;
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
            if _job_launch_in_grace(job) {
                if not ("control" in job) or not job["control"] {
                    job["control"] = "running";
                    changed = True;
                }
                if not ("active" in job) or not job["active"] {
                    job["active"] = True;
                    changed = True;
                }
                if _refresh_running_job_state(project_name, job_key, job) {
                    changed = True;
                }
                continue;
            }
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
            optimize_result_status = "";
            if job_key == "optimize" and "log" in job and job["log"] {
                optimize_result_status = _read_optimize_result_status(str(job["log"]));
            }
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
                refresh_metrics = True;
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
                refresh_metrics = True;
            } elif job_key == "optimize"
            and (
                _has_optimize_outputs_after(project_name, started_at)
                or optimize_result_status == "improved"
                or optimize_result_status == "no_improvement"
            ) {
                job["status"] = "completed";
                job["control"] = "idle";
                job["active"] = False;
                job["phase"] = "done";
                job["progress_percent"] = 1.0;
                job["result"] = "completed";
                if optimize_result_status == "no_improvement" {
                    job["result"] = "completed_no_improvement";
                    job["message"] = "Optimization finished; no valid improvements this run.";
                } elif (
                    not ("message" in job)
                    or not job["message"]
                    or str(job["message"]).strip().lower().startswith("queued for optimization")
                ) {
                    job["message"] = "Kernel optimization completed.";
                }
                refresh_metrics = True;
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
                refresh_metrics = True;
            } else {
                job["status"] = "error";
                job["control"] = "idle";
                job["active"] = False;
                job["phase"] = "error";
                job["result"] = "error";
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
    if refresh_metrics {
        try {
            _refresh_project_metrics(project_name);
        } except Exception as e {
            print(f"Failed to refresh project metrics: {e}");
        }
    }
    return state;
}

def:pub _read_state(project_name: str) -> dict {
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

def:pub _update_job_state(project_name: str, job_key: str, updates: dict) -> None {
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
    status_text = (
        str(job_state["status"]).strip().lower()
        if "status" in job_state and job_state["status"]
        else ""
    );
    if (
        status_text == "completed" or status_text == "success"
    ) and (
        job_key == "profile"
        or job_key == "generate"
        or job_key == "optimize"
        or job_key == "benchmark"
    ) {
        try {
            _refresh_project_metrics(project_name);
        } except Exception as e {
            print(f"Failed to refresh project metrics from job state update: {e}");
        }
    }
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
    # Avoid Linux-specific /proc probing on other platforms. Keep the existing Linux
    # behavior when procfs is available so zombies are treated as not alive there.
    try {
        if sys.platform.startswith("linux") {
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

def _job_launch_in_grace(job: dict, grace_seconds: float = 15.0) -> bool {
    if not job {
        return False;
    }
    recent_at = (
        str(job["updated_at"])
        if "updated_at" in job and job["updated_at"]
        else (
            str(job["started_at"])
            if "started_at" in job and job["started_at"]
            else ""
        )
    );
    if not recent_at {
        return False;
    }
    try {
        age_seconds = (
            datetime.now() - datetime.fromisoformat(recent_at)
        ).total_seconds();
        return age_seconds >= 0 and age_seconds < grace_seconds;
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

def:pub _modified_after(file_path: str, started_at: str) -> bool {
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

def:pub _has_generate_outputs_after(project_name: str, started_at: str = "") -> bool {
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

def:pub _has_optimize_outputs_after(project_name: str, started_at: str = "") -> bool {
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

def:pub _has_benchmark_outputs_after(project_name: str, started_at: str = "") -> bool {
    bench_path = path.join(
        _project_dir(project_name), "benchmarks", "op_benchmarks.json"
    );
    return _modified_after(bench_path, started_at);
}

def:pub _read_log_excerpt(log_path: str, max_bytes: int = 131072) -> str {
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

def:pub _extract_job_error_summary(log_path: str) -> str {
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
            or "failed to extract code" in line_lower
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
        line_lower = line.lower();
        if line and not line.startswith("[job]") {
            if line_lower.startswith("[workflow] running:") {
                idx = idx - 1;
                continue;
            }
            if line_lower.startswith("[optimize-result]") {
                idx = idx - 1;
                continue;
            }
            if line_lower.startswith("[workflow-optimize-result]") {
                idx = idx - 1;
                continue;
            }
            return line;
        }
        idx = idx - 1;
    }
    return "";
}

def _read_optimize_result_status(log_path: str) -> str {
    log_text = _read_log_excerpt(log_path);
    if not log_text {
        return "";
    }
    lines = log_text.splitlines();
    idx = len(lines) - 1;
    while idx >= 0 {
        line_lower = str(lines[idx]).strip().lower();
        if (
            ("[optimize-result]" in line_lower or "[workflow-optimize-result]" in line_lower)
            and "status=" in line_lower
        ) {
            if "status=no_improvement" in line_lower {
                return "no_improvement";
            }
            if "status=improved" in line_lower {
                return "improved";
            }
            if "status=hard_error" in line_lower {
                return "hard_error";
            }
        }
        idx = idx - 1;
    }
    log_lower = log_text.lower();
    if (
        "status: false" in log_lower
        and not ("traceback" in log_lower)
        and not ("[system error]" in log_lower)
        and not ("[worker crash]" in log_lower)
        and not ("[compilation failed]" in log_lower)
        and not ("[runtime error" in log_lower)
    ) {
        return "no_improvement";
    }
    return "";
}

def _running_job_progress(job: dict) -> float {
    if not job {
        return 0.0;
    }
    if "progress" in job
    and job["progress"]
    and "percent" in job["progress"]
    and job["progress"]["percent"] is not None {
        try {
            return float(job["progress"]["percent"]);
        } except Exception {
            # fall back to progress_percent below
        }
    }
    if "progress_percent" in job and job["progress_percent"] is not None {
        try {
            return float(job["progress_percent"]);
        } except Exception {
            return 0.0;
        }
    }
    return 0.0;
}

def _finalize_running_job_if_ready(project_name: str, job_key: str, job: dict) -> bool {
    if not job {
        return False;
    }
    started_at = (
        str(job["started_at"])
        if "started_at" in job and job["started_at"]
        else ""
    );
    progress_complete = _running_job_progress(job) >= 0.999;
    if not progress_complete {
        return False;
    }

    completed = False;
    if job_key == "profile" {
        completed = _has_profile_outputs(project_name);
        if completed {
            if not ("message" in job)
            or not job["message"]
            or str(job["message"]).strip().lower() == "finalizing results" {
                job["message"] = "Profile and baseline benchmark completed.";
            }
        }
    } elif job_key == "generate" {
        completed = (
            _has_generate_outputs_after(project_name, started_at)
            or _has_optimize_outputs_after(project_name, started_at)
            or _has_benchmark_outputs_after(project_name, started_at)
        );
        if completed and (not ("message" in job) or not job["message"]) {
            job["message"] = "Kernel generation completed.";
        }
    } elif job_key == "optimize" {
        completed = _has_optimize_outputs_after(project_name, started_at);
        if completed and (not ("message" in job) or not job["message"]) {
            job["message"] = "Kernel optimization completed.";
        }
    } elif job_key == "benchmark" {
        completed = _has_benchmark_outputs_after(project_name, started_at);
        if completed and (not ("message" in job) or not job["message"]) {
            job["message"] = "Benchmark completed.";
        }
    }

    if not completed {
        return False;
    }

    job["status"] = "completed";
    job["control"] = "idle";
    job["active"] = False;
    job["phase"] = "done";
    job["progress_percent"] = 1.0;
    job["updated_at"] = datetime.now().isoformat();
    return True;
}

def:pub _refresh_running_job_state(project_name: str, job_key: str, job: dict) -> bool {
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
    if _finalize_running_job_if_ready(project_name, job_key, job) {
        changed = True;
    }
    if changed {
        job["updated_at"] = datetime.now().isoformat();
    }
    return bool(changed);
}

def _default_queue_state() -> dict {
    return {
        "active_tasks": {},
        "pending_operators": [],
        "current_operator": "",
        "benchmark_slot": {"now": None, "pending": []},
        "job_queue": [],
        "launching_job": {}
    };
}

def _launching_job_active(queue_state: dict, grace_seconds: float = 30.0) -> bool {
    if not queue_state {
        return False;
    }
    launch = (
        queue_state["launching_job"]
        if "launching_job" in queue_state and queue_state["launching_job"]
        else {}
    );
    if not launch {
        return False;
    }
    started_at = (
        str(launch["started_at"])
        if "started_at" in launch and launch["started_at"]
        else ""
    );
    if not started_at {
        return False;
    }
    try {
        age_seconds = (
            datetime.now() - datetime.fromisoformat(started_at)
        ).total_seconds();
        return age_seconds >= 0 and age_seconds < grace_seconds;
    } except Exception {
        return False;
    }
}

def:pub _mark_job_launching(
    project_name: str,
    job_key: str,
    ops_list: list = [],
    tag: str = ""
) -> None {
    q_state = _read_queue_state(project_name);
    q_state["launching_job"] = {
        "job_key": job_key,
        "ops_list": [str(op) for op in ops_list if op],
        "tag": str(tag) if tag else "",
        "started_at": datetime.now().isoformat()
    };
    _write_queue_state(project_name, q_state);
    _append_queue_debug(
        project_name,
        "launch",
        "marked job launching",
        {
            "job_key": job_key,
            "ops_list": [str(op) for op in ops_list if op],
            "tag": str(tag) if tag else ""
        }
    );
}

def:pub _clear_job_launching(project_name: str, job_key: str = "") -> None {
    q_state = _read_queue_state(project_name);
    launch = (
        q_state["launching_job"]
        if "launching_job" in q_state and q_state["launching_job"]
        else {}
    );
    if not launch {
        return;
    }
    launch_key = str(launch["job_key"]) if "job_key" in launch and launch["job_key"] else "";
    if job_key and launch_key and launch_key != str(job_key) {
        return;
    }
    q_state["launching_job"] = {};
    _write_queue_state(project_name, q_state);
    _append_queue_debug(
        project_name,
        "launch",
        "cleared launching marker",
        {"job_key": launch_key}
    );
}

def _append_queue_debug(
    project_name: str,
    source: str,
    message: str,
    payload: dict = {}
) -> None {
    try {
        logs_dir = path.join(_project_dir(project_name), "logs");
        _ensure_dir(logs_dir);
        log_path = path.join(logs_dir, "queue_debug.log");
        line = (
            datetime.now().isoformat()
            + " | "
            + str(source)
            + " | "
            + str(message)
        );
        if payload and len(payload) > 0 {
            line = line + " | " + dumps(payload);
        }
        with open(log_path, "a") as f {
            f.write(line + "\n");
        }
    } except Exception as e {
        print(f"Failed to append queue debug log: {e}");
    }
}

def _terminal_queue_tasks(active_tasks: dict) -> dict {
    preserved = {};
    if not active_tasks {
        return preserved;
    }
    for task_key in active_tasks {
        task = active_tasks[task_key];
        if not task {
            continue;
        }
        current_step = (
            str(task["current_step"])
            if "current_step" in task and task["current_step"]
            else ""
        );
        if current_step == "Done" or current_step == "Failed" {
            preserved[str(task_key)] = task;
        }
    }
    return preserved;
}

def:pub _read_queue_state(project_name: str) -> dict {
    queue_path = path.join(_project_dir(project_name), "queue.json");
    state = _default_queue_state();
    if not path.exists(queue_path) {
        return state;
    }
    try {
        with open(queue_path, "r") as f {
            raw = load(f);
        }
        if raw {
            for key in raw {
                state[key] = raw[key];
            }
        }
    } except Exception as e {
        print(f"Failed to read queue state: {e}");
    }
    if not ("job_queue" in state) or not state["job_queue"] {
        state["job_queue"] = [];
    }
    if not ("launching_job" in state) or not state["launching_job"] {
        state["launching_job"] = {};
    }
    return state;
}

def:pub _write_queue_state(project_name: str, queue_state: dict) -> None {
    queue_path = path.join(_project_dir(project_name), "queue.json");
    state = _default_queue_state();
    if queue_state {
        for key in queue_state {
            state[key] = queue_state[key];
        }
    }
    if not ("job_queue" in state) or not state["job_queue"] {
        state["job_queue"] = [];
    }
    if not ("launching_job" in state) or not state["launching_job"] {
        state["launching_job"] = {};
    }
    try {
        tmp_path = (
            queue_path
            + "."
            + str(os.getpid())
            + "."
            + str(int(datetime.now().timestamp() * 1000))
            + ".tmp"
        );
        with open(tmp_path, "w") as f {
            dump(state, f, indent=2);
        }
        os.replace(tmp_path, queue_path);
    } except Exception as e {
        print(f"Failed to write queue state: {e}");
    }
}

def _job_is_active(job: dict) -> bool {
    if not job {
        return False;
    }
    if "active" in job and job["active"] {
        return True;
    }
    status_text = (
        str(job["status"]).strip().lower()
        if "status" in job and job["status"]
        else ""
    );
    return (
        status_text == "running"
        or status_text == "queued"
        or status_text == "paused"
    );
}

def _queue_has_running_work(queue_state: dict) -> bool {
    if not queue_state {
        return False;
    }
    if _launching_job_active(queue_state) {
        return True;
    }
    if "current_operator" in queue_state and queue_state["current_operator"] {
        return True;
    }
    active_tasks = queue_state["active_tasks"] if "active_tasks" in queue_state and queue_state["active_tasks"] else {};
    for task_key in active_tasks {
        task = active_tasks[task_key];
        current_step = (
            str(task["current_step"])
            if task and "current_step" in task and task["current_step"]
            else ""
        );
        if current_step != "Done" and current_step != "Failed" {
            return True;
        }
    }
    return False;
}

def _queue_has_backlog(queue_state: dict) -> bool {
    if not queue_state {
        return False;
    }
    if "job_queue" in queue_state and queue_state["job_queue"] and len(queue_state["job_queue"]) > 0 {
        return True;
    }
    return False;
}

def:pub _get_active_project_job(project_name: str, include_backlog: bool = True) -> dict {
    state = _read_state(project_name);
    for job_key in ["profile", "generate", "optimize", "benchmark"] {
        job = state[job_key] if state and job_key in state else {};
        if _job_is_active(job) {
            return {
                "jobKey": job_key,
                "message": f"{job_key} job already in progress."
            };
        }
    }
    queue_state = _read_queue_state(project_name);
    if _queue_has_running_work(queue_state) {
        return {
            "jobKey": "queue",
            "message": "project work already in progress."
        };
    }
    if include_backlog and _queue_has_backlog(queue_state) {
        return {
            "jobKey": "queue",
            "message": "project queue already has pending jobs."
        };
    }
    return {};
}

def _apply_job_queue_state(project_name: str, job_request: dict) -> None {
    job_key = str(job_request["job_key"]) if "job_key" in job_request and job_request["job_key"] else "";
    tag = str(job_request["tag"]) if "tag" in job_request and job_request["tag"] else "";
    ops_list = [];
    raw_ops = job_request["ops_list"] if "ops_list" in job_request and job_request["ops_list"] else [];
    for op in raw_ops {
        if op {
            ops_list.append(str(op));
        }
    }
    if job_key == "generate" {
        _write_initial_queue_state(project_name, ops_list, tag if tag else "[GEN]");
        return;
    }
    if job_key != "optimize" {
        return;
    }

    workers = 1;
    try {
        workers = int(job_request["workers"]) if "workers" in job_request else 1;
    } except Exception {
        workers = 1;
    }
    if workers <= 1 {
        _write_initial_queue_state(project_name, ops_list, tag if tag else "[OPT]");
        return;
    }

    q_state = _read_queue_state(project_name);
    q_state["active_tasks"] = _terminal_queue_tasks(
        q_state["active_tasks"] if "active_tasks" in q_state else {}
    );
    q_state["benchmark_slot"] = {"now": None, "pending": []};
    q_state["current_operator"] = str(ops_list[0]) if len(ops_list) > 0 else "";
    q_state["pending_operators"] = [str(o) for o in ops_list];
    q_state["launching_job"] = {};
    _write_queue_state(project_name, q_state);
}

def:pub _enqueue_project_job(project_name: str, job_request: dict) -> dict {
    q_state = _read_queue_state(project_name);
    job_queue = q_state["job_queue"] if "job_queue" in q_state and q_state["job_queue"] else [];

    request_copy = {};
    for key in job_request {
        request_copy[key] = job_request[key];
    }
    queue_id = (
        str(request_copy["job_key"]) + ":" + str(int(datetime.now().timestamp() * 1000))
    );
    request_copy["queue_id"] = queue_id;
    request_copy["enqueued_at"] = datetime.now().isoformat();
    job_queue.append(request_copy);
    q_state["job_queue"] = job_queue;
    _write_queue_state(project_name, q_state);
    _append_queue_debug(
        project_name,
        "enqueue",
        "queued job",
        {
            "job_key": request_copy["job_key"] if "job_key" in request_copy else "",
            "queue_id": queue_id,
            "queue_len": len(job_queue),
            "active_tasks": (
                len(q_state["active_tasks"])
                if "active_tasks" in q_state and q_state["active_tasks"]
                else 0
            ),
            "current_operator": (
                q_state["current_operator"]
                if "current_operator" in q_state
                else ""
            )
        }
    );
    return {"position": len(job_queue), "job": request_copy};
}

def:pub _queue_or_start_job(
    project_name: str,
    job_key: str,
    cmd: list,
    log_name: str,
    job_request: dict = {}
) -> dict {
    active_job = _get_active_project_job(project_name);
    request = {};
    for key in job_request {
        request[key] = job_request[key];
    }
    request["job_key"] = job_key;
    request["cmd"] = cmd;
    request["log_name"] = log_name;
    _append_queue_debug(
        project_name,
        "queue_or_start",
        "received request",
        {
            "job_key": job_key,
            "active_job": active_job["jobKey"] if active_job and "jobKey" in active_job else "",
            "ops_list": request["ops_list"] if "ops_list" in request else []
        }
    );

    if active_job {
        queued = _enqueue_project_job(project_name, request);
        if "jobKey" in active_job and active_job["jobKey"] == "queue" {
            _start_next_queued_job(project_name);
        }
        _append_queue_debug(
            project_name,
            "queue_or_start",
            "return queued",
            {
                "job_key": job_key,
                "position": queued["position"] if "position" in queued else 0
            }
        );
        return {
            "success": True,
            "queued": True,
            "reason": "queued",
            "message": (
                f"Queued {job_key} job behind "
                + str(active_job["jobKey"] if "jobKey" in active_job else "active")
                + "."
            ),
            "position": queued["position"] if "position" in queued else 1,
            "jobKey": job_key
        };
    }

    _mark_job_launching(
        project_name,
        job_key,
        request["ops_list"] if "ops_list" in request else [],
        str(request["tag"]) if "tag" in request and request["tag"] else ""
    );
    try {
        result = _start_job(project_name, job_key, cmd, log_name);
        if result
        and "success" in result and result["success"]
        and not ("skipped" in result and result["skipped"]) {
            _apply_job_queue_state(project_name, request);
        }
        _append_queue_debug(
            project_name,
            "queue_or_start",
            "return start result",
            {
                "job_key": job_key,
                "success": bool(result["success"]) if result and "success" in result else False,
                "skipped": bool(result["skipped"]) if result and "skipped" in result else False,
                "pid": result["pid"] if result and "pid" in result else None
            }
        );
        return result;
    } finally {
        _clear_job_launching(project_name, job_key);
    }
}

def:pub _start_next_queued_job(project_name: str) -> dict {
    active_job = _get_active_project_job(project_name, False);
    if active_job {
        return {
            "success": False,
            "reason": "busy",
            "jobKey": active_job["jobKey"] if "jobKey" in active_job else ""
        };
    }

    q_state = _read_queue_state(project_name);
    job_queue = q_state["job_queue"] if "job_queue" in q_state and q_state["job_queue"] else [];
    if len(job_queue) == 0 {
        return {"success": False, "reason": "empty"};
    }

    next_job = job_queue[0];
    next_key = str(next_job["job_key"]) if "job_key" in next_job and next_job["job_key"] else "";
    next_cmd = next_job["cmd"] if "cmd" in next_job and next_job["cmd"] else [];
    next_log = str(next_job["log_name"]) if "log_name" in next_job and next_job["log_name"] else (next_key + ".log");
    if not next_key or len(next_cmd) == 0 {
        return {"success": False, "reason": "invalid_job"};
    }

    result = _start_job(project_name, next_key, next_cmd, next_log);
    if result
    and "success" in result and result["success"]
    and not ("skipped" in result and result["skipped"]) {
        remaining = [];
        idx = 0;
        for queued_job in job_queue {
            if idx == 0 {
                idx = idx + 1;
                continue;
            }
            remaining.append(queued_job);
            idx = idx + 1;
        }
        q_state["job_queue"] = remaining;
        _write_queue_state(project_name, q_state);
        _apply_job_queue_state(project_name, next_job);
        result["queued_from_queue"] = True;
        _append_queue_debug(
            project_name,
            "start_next",
            "started queued job",
            {
                "job_key": next_key,
                "queue_remaining": len(remaining),
                "pid": result["pid"] if "pid" in result else None
            }
        );
    } else {
        _append_queue_debug(
            project_name,
            "start_next",
            "failed to start queued job",
            {
                "job_key": next_key,
                "success": bool(result["success"]) if result and "success" in result else False,
                "skipped": bool(result["skipped"]) if result and "skipped" in result else False
            }
        );
    }
    return result;
}

def:pub _start_job(project_name: str, job_key: str, cmd: list, log_name: str) -> dict {
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
    and (_pid_is_alive(current_pid) or _job_launch_in_grace(current)) {
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
    child_env["KFORGE_STATE_PATH"] = path.join(_project_dir(project_name), "state.json");
    child_env["KFORGE_JOB_KEY"] = job_key;
    child_env["PYTHONUNBUFFERED"] = "1";
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

def:pub _write_initial_queue_state(project_name: str, ops_list: list, tag: str) -> None {
    active_tasks = {};
    if tag == "[OPT]" {
        # Key must match pipeline's task_key = f"seq_opt_{op_name}" so the
        # pipeline's first update_queue_state merges into this entry rather
        # than creating a separate task that duplicates it in the UI.
        for op in ops_list {
            active_tasks["seq_opt_" + str(op)] = {
                "tag": "[OPT]",
                "op_name": str(op),
                "current_step": "Generating",
                "attempt_current": 1,
                "attempt_max": 3,
                "status": "In Progress"
            };
        }
    } else {
        for op in ops_list {
            active_tasks["gen_" + str(op)] = {
                "tag": "[GEN]",
                "op_name": str(op),
                "current_step": "Monitoring",
                "attempt_current": 1,
                "attempt_max": 8,
                "status": "Starting"
            };
        }
    }
    queue_data = _read_queue_state(project_name);
    merged_tasks = _terminal_queue_tasks(
        queue_data["active_tasks"] if "active_tasks" in queue_data else {}
    );
    for task_key in active_tasks {
        merged_tasks[str(task_key)] = active_tasks[task_key];
    }
    queue_data["pending_operators"] = [str(o) for o in ops_list];
    queue_data["active_tasks"] = merged_tasks;
    queue_data["benchmark_slot"] = {"now": None, "pending": []};
    queue_data["current_operator"] = str(ops_list[0]) if len(ops_list) > 0 else "";
    queue_data["launching_job"] = {};
    _write_queue_state(project_name, queue_data);
}

def:pub _start_job_container(
    project_name: str, job_key: str, cmd: list, log_name: str, image: str
) -> dict {
    return _start_job(project_name, job_key, cmd, log_name);
}

def:pub _find_optimized_dir(project_name: str) -> str {
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

# --- Metrics computation (lives here to avoid circular imports with optimization_results) ---

def:pub _normalize_metric_op_name(raw: Any) -> str {
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

def _compute_project_metrics(project_name: str) -> dict {
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
        "errors": bench_errors,
        "metrics_version": 0
    };
    return computed_payload;
}

def:pub _refresh_project_metrics(project_name: str) -> dict {
    project_dir = _project_dir(project_name);
    computed_payload = _compute_project_metrics(project_name);
    _upsert_project_catalog(project_name, project_dir);
    _upsert_project_metric_rows(project_name, computed_payload);
    persisted = _read_project_metric_rows(project_name);
    if persisted and "metrics_version" in persisted {
        return persisted;
    }
    return computed_payload;
}

def:pub _collect_project_metrics(project_name: str) -> dict {
    _upsert_project_catalog(project_name, _project_dir(project_name));
    persisted = _read_project_metric_rows(project_name);
    if persisted {
        version_val = 0;
        if "metrics_version" in persisted and persisted["metrics_version"] is not None {
            version_val = _to_int(persisted["metrics_version"]);
        }
        rows_len = (
            len(persisted["rows"])
            if "rows" in persisted and persisted["rows"]
            else 0
        );
        if version_val > 0 or rows_len > 0 {
            persisted_updated_at = (
                str(persisted["updated_at"])
                if "updated_at" in persisted and persisted["updated_at"]
                else ""
            );
            summary_path = path.join(_project_dir(project_name), "io", "summary.json");
            needs_refresh = False;
            if persisted_updated_at {
                if _modified_after(summary_path, persisted_updated_at) {
                    needs_refresh = True;
                } elif _has_benchmark_outputs_after(project_name, persisted_updated_at) {
                    needs_refresh = True;
                } elif _has_optimize_outputs_after(project_name, persisted_updated_at) {
                    needs_refresh = True;
                }
            }
            if needs_refresh {
                return _refresh_project_metrics(project_name);
            }
            return persisted;
        }
    }
    return _refresh_project_metrics(project_name);
}
