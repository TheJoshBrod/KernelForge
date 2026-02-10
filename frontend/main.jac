"""Main entry point for frontend."""
        
import from json { load, loads, dump }
import from pathlib { Path }
import from base64 { b64decode, b64encode }
import from os { path, makedirs, listdir }
import from shutil { rmtree }
import from datetime { datetime }
import socket;
import paramiko;
import torch;

glob config_store: dict = {
         "api_keys": {"openai": "", "gemini": "", "anthropic": ""},
         "selected_gpu_index": 0
     };

glob config_path: str = Path("config.json");

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

walker:pub CreateProject {
    has projectName: str;
    has code: str;
    has weightsFileB64: str = "";
    has validationSetPath: str = "";

    can create with `root entry {
        try {
            # Define projects directory
            projects_dir = "projects";
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
            if self.code {
                with open(path.join(project_path, "model.py"), "w") as f {
                    f.write(self.code);
                }
            }

            # Save weights.pt
            if self.weightsFileB64 {
                weights_data = b64decode(self.weightsFileB64);
                with open(path.join(project_path, "weights.pt"), "wb") as f {
                    f.write(weights_data);
                }
            }

            # Save config.json
            config = {
                "validation_set": self.validationSetPath
                if self.validationSetPath
                else None,
                "created_at": datetime.now().isoformat()
            };
            with open(path.join(project_path, "config.json"), "w") as f {
                dump(config, f, indent=2);
            }

            report {"success": True, "name": self.projectName};
        } except Exception as e {
            print(f"Error creating project: {e}");
            report {"error": "Failed to create project", "success": False};
        }
    }
}

walker:pub DeleteProject {
    has projectName: str;

    can delete with `root entry {
        try {
            projects_dir = "projects";
            project_path = path.join(projects_dir, self.projectName);

            if path.exists(project_path) {
                rmtree(project_path);
                report {"success": True, "name": self.projectName};
            } else {
                report {"error": "Project not found", "success": False};
            }
        } except Exception as e {
            print(f"Error deleting project: {e}");
            report {"error": str(e), "success": False};
        }
    }
}

walker:pub GetProjects {
    can return_projects with `root entry {
        try {
            projects_dir = "projects";
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

walker:pub GetOptimizationData {
    has projectName: str;
    has opName: str;

    can return_data with `root entry {
        try {
            # HARDCODED PROJECT PATH AS REQUESTED, DYNAMIC OP NAME
            base_path = "kernels/optimized/NVIDIA_GeForce_GTX_1660_Ti_egr6dqhOEU";
            
            # Helper to check existence
            def check_exists(suffix: str) -> str {
                if path.exists(suffix) { return suffix; }
                up_one = path.join("..", suffix);
                if path.exists(up_one) { return up_one; }
                return "";
            }

            # 1. Try exact match
            target_suffix = path.join(base_path, self.opName, "nodes");
            nodes_dir = check_exists(target_suffix);
            
            # 2. Try torch_nn_functional_ prefix (common pattern)
            if not nodes_dir {
                 target_suffix_prefixed = path.join(base_path, f"torch_nn_functional_{self.opName}", "nodes");
                 nodes_dir = check_exists(target_suffix_prefixed);
            }

            if not nodes_dir {
                 print(f"Path not found for op: {self.opName} (checked exact and torch_nn_functional_ prefix)");
                 report {"nodes": []};
                 disengage;
            }

            print(f"Reading from: {nodes_dir}");

            nodes_data = [];
            if path.exists(nodes_dir) {
                files = listdir(nodes_dir);
                for file_name in files {
                    if file_name.endswith(".json") {
                        file_path = path.join(nodes_dir, file_name);
                        with open(file_path, "r") as f {
                            node_data_item = load(f);
                            nodes_data.append(node_data_item);
                        }
                    }
                }
            }
            
            report {"nodes": nodes_data};

        } except Exception as e {
            print(f"Error getting optimization data: {e}");
            report {"nodes": [], "error": str(e)};
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
            try {
                import torch;
                if torch.cuda.is_available() {
                    count = torch.cuda.device_count();
                    for i in range(count) {
                        device_name = torch.cuda.get_device_name(i);
                        props = torch.cuda.get_device_properties(i);
                        total_memory = props.total_memory / (1024 ** 3);
                        
                        arch_name = "Unknown";
                        if props.major == 7 { arch_name = "Volta/Turing"; }
                        elif props.major == 8 { arch_name = "Ampere"; }
                        elif props.major == 9 { arch_name = "Hopper"; }

                        all_gpus.append({
                            "name": f"{device_name} (Local)",
                            "memory": f"{total_memory:.1f} GB",
                            "compute_capability": f"v{props.major}.{props.minor} {arch_name}",
                            "sm_count": props.multi_processor_count
                        });
                    }
                }
            } except Exception as e {
                print(f"Local GPU check failed: {e}");
            }

            # 2. REMOTE CHECK VIA SSH
            if self.host and self.user {
                try {
                    client = paramiko.SSHClient();
                    client.set_missing_host_key_policy(paramiko.AutoAddPolicy());
                    
                    # Resolve key path
                    k_path = None;
                    if self.key_path {
                        if path.isdir(self.key_path) {
                            possible = ["id_rsa", "id_ed25519"];
                            for p in possible {
                                fp = path.join(self.key_path, p);
                                if path.exists(fp) { k_path = fp; break; }
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
            print(f"Attempting SSH connection to {self.user}@{self.host}:{self.port}...");
            
            # Connect using key_path if provided, otherwise password, or both (paramiko handles precedence usually)
            # We'll explicitly handle the key_filename argument logic
            k_path = None;
            if self.key_path {
                if path.isdir(self.key_path) {
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
                        print(f"Warning: {self.key_path} is a directory but no common keys found inside.");
                    }
                } else {
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
            exec_res = client.exec_command('echo "Connection Successful"');
            stdin = exec_res[0];
            stdout = exec_res[1];
            stderr = exec_res[2];
            exit_status = stdout.channel.recv_exit_status();
            
            if exit_status == 0 {
                output = stdout.read().decode().strip();
                print(f"Command execution successful: {output}");
                report {"success": True, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")};
            } else {
                error_msg = stderr.read().decode().strip();
                print(f"Command execution failed with status {exit_status}: {error_msg}");
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

cl {
    import from react { useState, useEffect }
    import from sonner { Toaster }
    import from context.ConfigContext { ConfigProvider }
    import from .components.Home { Home }
    import from .components.Project { Project }
    import from .components.Settings { Settings }
    import from .components.NewProject { NewProject }
    import from .components.OpDetails { OpDetails }

    import "./styles.css";

    def:pub app() -> any {
        has page: str = "/";
        has history: list = [];

        useEffect(
            lambda -> None {
                document.title = "CGinS";
                link = document.querySelector("link[rel*='icon']") or document.createElement("link");
                link.type = "image/svg+xml";
                link.rel = "shortcut icon";
                link.href = "data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>👻</text></svg>";
                document.head.appendChild(link);
            },
            []
        );

        # Client-side wrappers for walkers
        async def client_get_config() -> dict {
            resp = await root spawn GetConfig();
            return resp.reports[0];
        }

        async def client_save_config(cfg: any) -> bool {
            resp = await root spawn SaveConfig(cfg=cfg);
            return resp.reports[0];
        }

        async def client_create_project(
            projectName: str, code: str, weightsFileB64: str, validationSetPath: str
        ) -> dict {
            resp = await root spawn CreateProject(
                projectName=projectName,
                code=code,
                weightsFileB64=weightsFileB64,
                validationSetPath=validationSetPath
            );
            return resp.reports[0] if resp.reports else {};
        }

        async def client_delete_project(projectName: str) -> dict {
            resp = await root spawn DeleteProject(projectName=projectName);
            return resp.reports[0] if resp.reports else {};
        }

        async def client_get_projects() -> dict {
            resp = await root spawn GetProjects();
            return resp.reports[0] if resp.reports else {"projects": []};
        }

        async def client_get_optimization_data(projectName: str, opName: str) -> dict {
            console.log(f"Spawning GetOptimizationData for {projectName}, {opName}");
            resp = await root spawn GetOptimizationData(projectName=projectName, opName=opName);
            console.log("GetOptimizationData response:", resp);
            return resp.reports[0] if resp.reports else {"nodes": []};
        }

        async def client_get_system_info(host: str = "", user: str = "", password: str = "", key_path: str = "") -> dict {
            resp = await root spawn GetSystemInfo(host=host, user=user, password=password, key_path=key_path);
            return resp.reports[0] if resp.reports else {};
        }

        async def client_check_ssh_connection(host: str, user: str, password: str = "", key_path: str = "") -> dict {
            # Now passing all auth details to the walker
            resp = await root spawn CheckSshConnection(
                host=host, 
                user=user, 
                password=password, 
                key_path=key_path
            );
            if resp.reports {
                 return resp.reports[0];
            }
            return {"success": False, "error": "No response"};
        }

        def navigate(path: str) -> None {
            console.log(f"Navigating to {path}");
            if page != path {
                history.push(page);
            }
            page = path;
            # window.history.pushState({}, "", path);
        }

        def go_back() -> None {
            if history.length > 0 {
                prev_page = history.pop();
                console.log(f"Going back to {prev_page}");
                page = prev_page;
            } else {
                navigate("/");
            }
        }

        parts = page.split("/");
        # path is /project/name -> ["", "project", "name"]
        is_project = (parts[1] == "project") if parts.length > 1 else False;
        project_name = parts[2] if is_project and parts.length > 2 else "";

        is_op_details = is_project and parts.length > 3;
        op_name = parts[3] if is_op_details else "";

        return
            <ConfigProvider
                getConfig={client_get_config}
                saveConfig={client_save_config}
            >
                <div
                    className="bg-zinc-950 min-h-screen text-zinc-100 font-sans antialiased"
                >
                    {<Home
                        onNavigate={navigate}
                        getProjectsWalker={client_get_projects}
                        getSystemInfoWalker={client_get_system_info}
                        deleteProjectWalker={client_delete_project}
                    />
                    if page == "/" or page == ""
                    else <Settings 
                        onNavigate={navigate} 
                        onReturn={go_back} 
                        getSystemInfoWalker={client_get_system_info} 
                        checkSshConnectionWalker={client_check_ssh_connection}
                    />
                    if page == "/settings"
                    else <NewProject
                        onNavigate={navigate}
                        createProjectWalker={client_create_project}
                    />
                    if page == "/new-project"
                    else <OpDetails
                        projectName={project_name}
                        opName={op_name}
                        onNavigate={navigate}
                        getOpDataWalker={client_get_optimization_data}
                    />
                    if is_op_details
                    else <Project name={project_name} onNavigate={navigate} />
                    if is_project
                    else <div
                        className="flex min-h-screen flex-col items-center justify-center"
                    >
                        <h1>
                            404 Not Found: {page}
                        </h1>
                        <button
                            onClick={lambda -> None { navigate("/");}}
                            className="mt-4 p-2 bg-zinc-800 rounded"
                        >
                            Back to Home
                        </button>
                    </div>}
                    <Toaster richColors={True} theme="dark" />
                </div>
            </ConfigProvider>;
    }
}
