"""Walkers that start or control running optimization jobs."""

import from json { load, dump }
import from os { path, listdir }
import sys;

import from .globals { repo_root, config_store }
import from .catalog_db { _ensure_dir }
import from .job_supervisor {
    _project_dir,
    _read_state,
    _update_job_state,
    _start_job,
    _queue_or_start_job,
    _append_queue_debug,
    _read_log_excerpt,
    _detect_hardware
}

def _resolve_current_gpu_info() -> Any {
    gpu_info = config_store["selected_gpu_info"];
    if not gpu_info {
        hw = _detect_hardware();
        gpus = hw["gpus"] if hw and "gpus" in hw and hw["gpus"] else [];
        if len(gpus) > 0 {
            gpu_info = gpus[0];
        }
    }
    return gpu_info;
}

def _write_benchmarked_gpu(project_name: str, gpu_info: Any) -> None {
    if not gpu_info {
        return;
    }
    cfg_path = path.join(_project_dir(project_name), "config.json");
    try {
        cfg = {};
        if path.exists(cfg_path) {
            with open(cfg_path, "r") as f {
                cfg = load(f);
            }
        }
        gpu_uuid = str(gpu_info["uuid"]) if "uuid" in gpu_info and gpu_info["uuid"] else "";
        gpu_name = str(gpu_info["name"]) if "name" in gpu_info and gpu_info["name"] else "Unknown";
        gpu_conn = str(gpu_info["connection"]) if "connection" in gpu_info and gpu_info["connection"] else "local";
        identity = gpu_uuid if gpu_uuid else f"{gpu_name}:{gpu_conn}";
        cfg["benchmarked_gpu"] = {
            "uuid": gpu_uuid,
            "name": gpu_name,
            "connection": gpu_conn,
            "identity": identity
        };
        with open(cfg_path, "w") as f {
            dump(cfg, f, indent=2);
        }
    } except Exception as e {
        print(f"Failed to write benchmarked_gpu for {project_name}: {e}");
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
        targetDevice: str = "",
        ssh_connections: list = [],
        active_ssh_index: int = -1,
        llmModel: str = "",
        llmProvider: str = "",
        workers: int = 1;

    can run with entry {
        _append_queue_debug(
            self.projectName,
            "StartGenerate",
            "walker entry",
            {"ops": self.ops}
        );
        ops_list = [];
        for o in self.ops {
            if o {
                ops_list.append(str(o));
            }
        }
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
                    if project_cfg and "backend" in project_cfg and project_cfg["backend"] {
                        target_device = str(project_cfg["backend"]).lower();
                    } elif project_cfg
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
        gen_gpu_info = _resolve_current_gpu_info();
        if self.benchmark {
            _write_benchmarked_gpu(self.projectName, gen_gpu_info);
        }
        result = _queue_or_start_job(
            self.projectName,
            "generate",
            cmd,
            "generate.log",
            {
                "ops_list": ops_list,
                "tag": "[GEN]",
                "workers": 1
            }
        );
        if result {
            gen_cc = str(gen_gpu_info["compute_capability"])
                if gen_gpu_info and "compute_capability" in gen_gpu_info
                else "";
            if not gen_gpu_info or gen_cc in ["0.0", "N/A", ""] {
                result["gpu_warning"] = "GPU specs could not be fully detected. LLM optimization prompts may use degraded hardware context. Check pycuda/NVML availability.";
            }
        }
        _append_queue_debug(
            self.projectName,
            "StartGenerate",
            "walker report",
            result if result else {}
        );
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
        _append_queue_debug(
            self.projectName,
            "StartOptimize",
            "walker entry",
            {"ops": self.ops, "workers": self.workers}
        );
        ops_list = [];
        for o in self.ops {
            if o {
                ops_list.append(str(o));
            }
        }
        io_dir = path.join(_project_dir(self.projectName), "io", "individual_ops");
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
        cmd.append("--workers");
        cmd.append(str(self.workers));
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
        opt_gpu_info = _resolve_current_gpu_info();
        _write_benchmarked_gpu(self.projectName, opt_gpu_info);
        result = _queue_or_start_job(
            self.projectName,
            "optimize",
            cmd,
            "optimize.log",
            {
                "ops_list": ops_list,
                "tag": "[OPT]",
                "workers": self.workers
            }
        );
        if result {
            opt_cc = str(opt_gpu_info["compute_capability"])
                if opt_gpu_info and "compute_capability" in opt_gpu_info
                else "";
            if not opt_gpu_info or opt_cc in ["0.0", "N/A", ""] {
                result["gpu_warning"] = "GPU specs could not be fully detected. LLM optimization prompts may use degraded hardware context. Check pycuda/NVML availability.";
            }
        }
        _append_queue_debug(
            self.projectName,
            "StartOptimize",
            "walker report",
            result if result else {}
        );
        report result;
    }
}

walker:pub StartBenchmark {
    has projectName: str;

    can run with entry {
        gpu_info = _resolve_current_gpu_info();
        _write_benchmarked_gpu(self.projectName, gpu_info);
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
        try {
            # Cancel any running optimizer/generator job first so the subprocess
            # stops calling update_queue_state() and re-populating the queue.
            job_state = _read_state(self.projectName);
            for job_key in ["optimize", "generate"] {
                job = job_state[job_key] if job_state and job_key in job_state else {};
                job_status = str(job["status"] if "status" in job else "").lower();
                if job_status == "running" or job_status == "queued" {
                    _update_job_state(self.projectName, job_key, {"control": "cancelled", "status": "cancelled"});
                }
            }
            # Clear the queue file.
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
