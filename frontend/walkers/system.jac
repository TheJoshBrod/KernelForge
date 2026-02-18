"""System and SSH walkers."""

import from json { loads }
import from base64 { b64encode }
import from os { path }
import from datetime { datetime }
import sys;
import subprocess;
import paramiko;
import torch;

import from .globals { repo_root, config_path, config_store }
import from json { load }

glob _hw_cache: dict = {};
glob _sys_info_cache: dict = {};
glob _sys_info_ts: float = 0.0;

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
        result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, timeout=10);
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

walker:pub GetSystemInfo {
    has host: str = "";
    has user: str = "";
    has password: str = "";
    has key_path: str = "";

    can return_info with entry {
        try {
            # Check cache (1 minute TTL)
            now_ts = datetime.now().timestamp();
            if _sys_info_cache and (now_ts - _sys_info_ts) < 60 {
                report _sys_info_cache ;
                disengage;
            }

            all_gpus = [];

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
                                "name": device_name,
                                "connection": "local",
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

            if self.host and self.user {
                try {
                    client = paramiko.SSHClient();
                    client.set_missing_host_key_policy(paramiko.AutoAddPolicy());

                    k_path = None;
                    if self.key_path {
                        self.key_path = path.expanduser(self.key_path);
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

                    cmd = f"""bash -lc 'python3 -c "import base64; exec(base64.b64decode(\\"{b64_script}\\"))"'""";

                    exec_res = client.exec_command(cmd);
                    stdout = exec_res[1];
                    stderr = exec_res[2];
                    exit_status = stdout.channel.recv_exit_status();

                    if exit_status == 0 {
                        out_json = stdout.read().decode().strip();
                        remote_gpus = loads(out_json);
                        for r_gpu in remote_gpus {
                            r_gpu["connection"] = self.host;
                            all_gpus.append(r_gpu);
                        }
                    } else {
                        err = stderr.read().decode().strip();
                        print(f"Remote python failed: {err}");
                    }
                    client.close();
                } except Exception as e {
                    print(f"SSH Check failed: {e}");
                }
            }

            result_payload = {
                "available": True if all_gpus else False,
                "gpus": all_gpus,
                "source": "mixed"
            };
            
            # Update cache
            _sys_info_cache = result_payload;
            _sys_info_ts = datetime.now().timestamp();

            report result_payload ;
        } except Exception as e {
            report {"available": False, "error": str(e)} ;
        }
    }
}

walker:pub CheckSshConnection {
    has host: str,
        user: str,
        password: str,
        key_path: str;

    can return_status with entry {
        try {
            client = paramiko.SSHClient();
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy());

            k_path = None;
            if self.key_path {
                self.key_path = path.expanduser(self.key_path);
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
                }
                report {
                    "success": True,
                    "gpus": remote_gpus,
                    "message": f"Connected to {self.host} successfully.",
                    "timestamp": datetime.now().isoformat()
                } ;
            } else {
                err = stderr.read().decode().strip();
                report {"success": False, "error": f"Remote script failed: {err}"} ;
            }
            client.close();
        } except Exception as e {
            report {"success": False, "error": f"Connection failed: {str(e)}"} ;
        }
    }
}

walker:pub TestLLMConnection {
    has cfg: dict = {};

    can run with entry {
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
        report result ;
    }
}

walker:pub DetectHardware {
    can detect with entry {
        global _hw_cache;
        if _hw_cache {
            report _hw_cache ;
        } else {
            try {
                code = "import json, torch; info={'cuda_available': torch.cuda.is_available(), 'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0, 'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else '', 'cuda_version': torch.version.cuda or '', 'mps_available': getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available(), 'mps_built': getattr(torch.backends, 'mps', None) and torch.backends.mps.is_built()}; print(json.dumps(info))";
                out = subprocess.check_output(
                    [sys.executable, "-c", code], cwd=repo_root, timeout=5
                );
                text = out.decode("utf-8").strip();
                if text {
                    _hw_cache = loads(text);
                    report _hw_cache ;
                } else {
                    report {"cuda_available": False, "mps_available": False} ;
                }
            } except Exception as e {
                report {
                    "cuda_available": False,
                    "mps_available": False,
                    "error": str(e)
                } ;
            }
        }
    }
}
