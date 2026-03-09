"""Walkers that start or control running optimization jobs."""

import from json { load, dump }
import from os { path, listdir }
import sys;

import from .globals { repo_root }
import from .catalog_db { _ensure_dir }
import from .job_supervisor {
    _project_dir,
    _read_state,
    _update_job_state,
    _start_job,
    _write_initial_queue_state,
    _read_log_excerpt
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
        targetDevice: str = "cuda",
        ssh_connections: list = [],
        active_ssh_index: int = -1,
        llmModel: str = "",
        llmProvider: str = "",
        workers: int = 1;

    can run with entry {
        # Get relevant directories for project 
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
        if target_device not in ["cuda", "mps", "metal", "triton", "cpu"] {
            report {"success": False, "error": f"Unsupported target device: {target_device}"};
            disengage;
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
        if self.workers > 1 {
            cmd.append("--workers");
            cmd.append(str(self.workers));
        }
        if self.llmModel {
            cmd.append("--llm-model");
            cmd.append(self.llmModel);
        }
        if self.llmProvider {
            cmd.append("--llm-provider");
            cmd.append(self.llmProvider);
        }
        if self.active_ssh_index >= 0
        and len(self.ssh_connections) > self.active_ssh_index {
            try {
                remote_cfg = {
                    "ssh_connections": self.ssh_connections,
                    "active_ssh_index": self.active_ssh_index
                };
                remote_cfg_path = path.join(
                    _project_dir(self.projectName), "ssh_config.json"
                );
                with open(remote_cfg_path, "w") as f {
                    dump(remote_cfg, f);
                }
                cmd.append("--remote");
                cmd.append(remote_cfg_path);
            } except Exception as e {
                print(f"Failed to write SSH config for remote generation: {e}");
            }
        }
        result = _start_job(self.projectName, "generate", cmd, "generate.log");
        if result and "success" in result and result["success"] {
            _write_initial_queue_state(self.projectName, ops_list, "[GEN]");
        }
        report result;
    }
}

walker:pub StartOptimize {
    has projectName: str,
        ops: list = [],
        iterations: int = 3,
        benchmark: bool = False,
        ssh_connections: list = [],
        active_ssh_index: int = -1,
        llm_model: str = "",
        llm_provider: str = "",
        workers: int = 4;

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
        # Removed unnecessary target device fallback for optimization
        cmd = [
            sys.executable,
            "-m",
            "src.optimizer.workflow",
            "optimize",
            "--project",
            self.projectName,
            "--ops",
            ",".join(ops_list)
        ];
        if self.iterations > 0 {
            cmd.append("--iterations");
            cmd.append(str(self.iterations));
        }
        if self.workers > 1 {
            cmd.append("--workers");
            cmd.append(str(self.workers));
        }
        if self.benchmark {
            cmd.append("--benchmark");
        }
        if self.llm_model {
            cmd.append("--llm-model");
            cmd.append(self.llm_model);
        }
        if self.llm_provider {
            cmd.append("--llm-provider");
            cmd.append(self.llm_provider);
        }
        if self.active_ssh_index >= 0
        and len(self.ssh_connections) > self.active_ssh_index {
            try {
                remote_cfg = {
                    "ssh_connections": self.ssh_connections,
                    "active_ssh_index": self.active_ssh_index
                };
                remote_cfg_path = path.join(
                    _project_dir(self.projectName), "ssh_config.json"
                );
                with open(remote_cfg_path, "w") as f {
                    dump(remote_cfg, f);
                }
                cmd.append("--remote");
                cmd.append(remote_cfg_path);
            } except Exception as e {
                print(f"Failed to write SSH config for remote optimization: {e}");
            }
        }
        result = _start_job(self.projectName, "optimize", cmd, "optimize.log");
        if result and "success" in result and result["success"] {
            if self.workers <= 1 {
                # Sequential: write placeholder with matching key so the pipeline
                # merges its first update into the same entry (no duplicate row).
                _write_initial_queue_state(self.projectName, ops_list, "[OPT]");
            } else {
                # Parallel: node IDs unknown until subprocess starts, so no
                # placeholder.  Clear any stale active_tasks from prior runs so
                # they don't appear alongside the new node entries.
                queue_path = path.join(_project_dir(self.projectName), "queue.json");
                try {
                    q_state = {};
                    if path.exists(queue_path) {
                        with open(queue_path, "r") as f {
                            q_state = load(f);
                        }
                    }
                    q_state["active_tasks"] = {};
                    q_state["benchmark_slot"] = {"now": None, "pending": []};
                    q_state["current_operator"] = str(ops_list[0]) if len(ops_list) > 0 else "";
                    q_state["pending_operators"] = [str(o) for o in ops_list];
                    with open(queue_path, "w") as f {
                        dump(q_state, f);
                    }
                } except Exception as e {
                    print(f"Failed to reset queue for parallel optimize: {e}");
                }
            }
        }
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

walker:pub ClearProjectQueue {
    has projectName: str;

    can run with entry {
        queue_path = path.join(_project_dir(self.projectName), "queue.json");
        active_steps = {"Monitoring", "Generating", "Validating", "Benchmarking", "Profiling"};
        try {
            state = {};
            if path.exists(queue_path) {
                with open(queue_path, "r") as f {
                    state = load(f);
                }
            }
            state["active_tasks"] = {};
            state["pending_operators"] = [];
            state["current_operator"] = "";
            state["benchmark_slot"] = {"now": None, "pending": []};
            with open(queue_path, "w") as f {
                dump(state, f);
            }
            report {"success": True};
        } except Exception as e {
            report {"success": False, "error": str(e)};
        }
    }
}

walker:pub GetJobLog {
    has projectName: str,
        jobKey: str,
        lines: int = 15;

    can run with entry {
        try {
            state = _read_state(self.projectName);
            job = state.get(self.jobKey, {}) if state else {};
            log_path = job.get("log", "") if job else "";
            if not log_path {
                report {"lines": [], "path": ""};
                disengage;
            }
            text = _read_log_excerpt(log_path, 8192);
            all_lines = [l for l in text.splitlines() if l.strip()];
            n = self.lines if self.lines > 0 else 15;
            tail = all_lines[-n:] if len(all_lines) > n else all_lines;
            report {"lines": tail, "path": log_path};
        } except Exception as e {
            report {"lines": [], "path": "", "error": str(e)};
        }
    }
}
