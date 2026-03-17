import os
import paramiko
import re
import struct
import pickle
import time
import traceback
from pathlib import Path
from src.optimizer.config.settings import settings, MIN_CUDA_VERSION


def _check_remote_nvcc(client: paramiko.SSHClient) -> None:
    """Raise RuntimeError if the remote machine's nvcc is missing or below the minimum version."""
    _, out, _ = execute_remote_command(client, "nvcc --version 2>&1 || echo NVCC_NOT_FOUND")
    if "NVCC_NOT_FOUND" in out or not out.strip():
        raise RuntimeError(
            "nvcc not found on the remote machine. "
            "Ensure CUDA is installed and nvcc is on the remote PATH."
        )
    match = re.search(r"release (\d+)\.(\d+)", out)
    if not match:
        raise RuntimeError(
            f"Could not parse remote nvcc version from output: {out!r}"
        )
    major, minor = int(match.group(1)), int(match.group(2))
    if (major, minor) < MIN_CUDA_VERSION:
        raise RuntimeError(
            f"Remote machine has CUDA {major}.{minor} which is too old. "
            f"Kernel Forge requires CUDA {MIN_CUDA_VERSION[0]}.{MIN_CUDA_VERSION[1]} or newer."
        )

REQUIRED_PACKAGES = [
    "torch",
    "numpy",
    "nvidia-ml-py",  # pynvml
    "pycuda"
]

SETUP_SCRIPT = """
import os, subprocess, sys, shutil

# Expand user to get home dir
WORKSPACE = os.path.expanduser("~/kforge_workspace")
VENV_PATH = os.path.join(WORKSPACE, "venv")

if not os.path.exists(WORKSPACE):
    print(f"DTO: Creating workspace at {WORKSPACE}")
    os.makedirs(WORKSPACE)

# Check if venv needs rebuild (version mismatch)
if os.path.exists(VENV_PATH):
    venv_python = os.path.join(VENV_PATH, "bin", "python")
    rebuild = False
    if not os.path.exists(venv_python):
        rebuild = True
    else:
        try:
            # Check major.minor version
            out = subprocess.check_output([venv_python, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"]).decode().strip()
            current = f"{sys.version_info.major}.{sys.version_info.minor}"
            if out != current:
                print(f"DTO: Venv version mismatch ({out} vs {current}). Rebuilding.")
                rebuild = True
        except Exception as e:
            print(f"DTO: Error checking venv version: {e}")
            rebuild = True
            
    if rebuild:
        print("DTO: Removing stale virtual environment...")
        shutil.rmtree(VENV_PATH)

# Create venv if missing
if not os.path.exists(VENV_PATH):
    print("DTO: Creating virtual environment...")
    subprocess.check_call([sys.executable, "-m", "venv", VENV_PATH])
    
    # Auto-install dependencies
    pip = os.path.join(VENV_PATH, "bin", "pip")
    print("DTO: Installing dependencies... (this may take a while)")
    subprocess.check_call([pip, "install", "torch", "numpy", "nvidia-ml-py", "pycuda"])
    print("DTO: Dependencies installed.")

print("DTO: Remote environment ready.")
"""

def connect_ssh(config: dict) -> paramiko.SSHClient:
    """
    Establishes an SSH connection based on the provided configuration.
    Config should match the structure used in SSHSettings.cl.jac:
    {
        "host": "...",
        "user": "...",
        "key_path": "..."   # optional
    }
    """
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connect_kwargs = {
        "hostname": config["host"],
        "username": config["user"],
        "timeout": settings.verifier_timeout_seconds, # Use timeout from settings or default
    }


    
    if config.get("key_path"):
        key_path = os.path.expanduser(config["key_path"])
        if os.path.exists(key_path):
            if os.path.isdir(key_path):
                # Try common key names if user provided a directory
                for key_name in ["id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"]:
                    candidate = os.path.join(key_path, key_name)
                    if os.path.exists(candidate):
                        connect_kwargs["key_filename"] = candidate
                        break
            else:
                connect_kwargs["key_filename"] = key_path

    try:
        print(f"DEBUG: Attempting SSH connection to {connect_kwargs['hostname']}...", flush=True)
        client.connect(**connect_kwargs)
        print("DEBUG: SSH Connected.", flush=True)
        return client
    except Exception as e:
        print(f"SSH Connection failed: {e}")
        raise e

def execute_remote_command(client: paramiko.SSHClient, command: str, stream_output: bool = False):
    """
    Executes a command on the remote server.
    """
    stdin, stdout, stderr = client.exec_command(command)
    
    exit_status = stdout.channel.recv_exit_status()
    out_str = stdout.read().decode().strip()
    err_str = stderr.read().decode().strip()

    return exit_status, out_str, err_str

def upload_files(client: paramiko.SSHClient, files: dict[str, str], remote_dir: str):
    """
    Uploads files to the remote directory.
    files: dict mapping local_path -> remote_filename (relative to remote_dir)
    """
    sftp = client.open_sftp()
    
    # Ensure remote directory exists recursively
    try:
        # Use exec_command to mkdir -p, which sftp doesn't support directly
        stdin, stdout, stderr = client.exec_command(f"mkdir -p {remote_dir}")
        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            print(f"Warning: Failed to create remote directory {remote_dir}: {stderr.read().decode()}")
    except Exception as e:
        print(f"Warning: SSH mkdir failed: {e}")

    for local_path, remote_name in files.items():
        remote_path = f"{remote_dir}/{remote_name}"
        sftp.put(local_path, remote_path)
    
    sftp.close()

def ensure_remote_setup(client: paramiko.SSHClient):
    """
    Checks for remote workspace and venv, creates them if missing.
    """
    # Execute the setup script via stdin
    print("DEBUG: Executing remote setup script...", flush=True)
    stdin, stdout, stderr = client.exec_command("python3.12 -")
    stdin.write(SETUP_SCRIPT)
    stdin.close()
    
    # Wait for completion
    print("DEBUG: Waiting for remote setup to complete...", flush=True)
    exit_status = stdout.channel.recv_exit_status()
    out = stdout.read().decode()
    err = stderr.read().decode()
    
    if exit_status != 0:
        print(f"Remote bootstrapping failed: {err}")
        print(f"Output: {out}")
        return False
        
    return True

class RemoteWorkerClient:
    def __init__(self, ssh_config: dict, worker_script_path: Path, aux_files: dict[str, str] = None):
        """
        Initializes the remote worker client.
        
        Args:
            ssh_config: SSH connection details (host, user, key_path).
            worker_script_path: Local path to the worker.py script to execute remotely.
            aux_files: Optional dict of {local_path: remote_filename} for additional dependencies (e.g. loader.py).
        """
        self.client = connect_ssh(ssh_config)

        # 1. Bootstrap
        if not ensure_remote_setup(self.client):
            raise RuntimeError("Failed to bootstrap remote environment")

        # 1b. Verify remote CUDA version before doing anything else
        _check_remote_nvcc(self.client)

        # 2. Upload worker script and dependencies
        files_to_upload = {
            str(worker_script_path): "worker.py"
        }
        if aux_files:
            files_to_upload.update(aux_files)
            
        print(f"DEBUG: Uploading worker from {worker_script_path}...", flush=True)
        upload_files(self.client, files_to_upload, "kforge_workspace")
        
        # 3. Start Worker Process
        # Use line buffered unbuffered python? -u is important
        start_cmd = "~/kforge_workspace/venv/bin/python3 -u ~/kforge_workspace/worker.py"
        print(f"DEBUG: Starting worker: {start_cmd}", flush=True)
        
        # We need a channel for interactive I/O
        self.transport = self.client.get_transport()
        self.channel = self.transport.open_session()
        self.channel.exec_command(start_cmd)
        
        self.stdin = self.channel.makefile('wb')
        self.stdout = self.channel.makefile('rb')
        self.stderr = self.channel.makefile_stderr('rb')
        
        # 4. Wait for READY signal
        print("DEBUG: Waiting for remote worker 'READY' signal...", flush=True)
        while True:
            line = self.stdout.readline().decode().strip()
            if not line:
                # EOF without READY
                err = self.stderr.read().decode()
                raise ConnectionError(f"Remote worker closed connection without READY signal. Error: {err}")
            
            if line == "READY":
                print("DEBUG: Remote worker is READY.", flush=True)
                break
            else:
                # Forward debug output from worker
                print(f"[REMOTE] {line}", flush=True)
            
    def send_task(self, command: str, data: dict = None):
        request = {'command': command, 'data': data or {}}
        payload = pickle.dumps(request)
        
        # Send Length + Data
        self.stdin.write(struct.pack('>I', len(payload)))
        self.stdin.write(payload)
        self.stdin.flush()
        
        # Read Response Length
        raw_len = self.stdout.read(4)
        if not raw_len:
            raise ConnectionError("Worker closed connection (EOF)")
            
        resp_len = struct.unpack('>I', raw_len)[0]
        
        # Read Response Data
        resp_data = self.stdout.read(resp_len)
        return pickle.loads(resp_data)
        
    def close(self):
        try:
            self.stdin.close()
            self.stdout.close()
            self.stderr.close()
            self.channel.close()
            self.client.close()
        except:
            pass
