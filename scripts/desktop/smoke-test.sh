#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python3 -u - <<'PY'
import json
import os
import re
import time
import urllib.request
from pathlib import Path


def detect_base_url() -> str:
    pattern = re.compile(
        r"kf-ui-server\.py --bind 127\.0\.0\.1 --port (\d+).*--api-base-url (http://127\.0\.0\.1:\d+)"
    )
    for _ in range(20):
        proc_text = os.popen(
            "pgrep -af 'kernel-forge-desktop|kf-ui-server.py|jac_client.plugin.src.targets.desktop.sidecar.main'"
        ).read()
        match = pattern.search(proc_text)
        if match:
            return f"http://127.0.0.1:{match.group(1)}"
        time.sleep(0.5)
    raise SystemExit(
        "Could not find a running desktop UI server. Start kernel-forge-desktop first."
    )


BASE_URL = detect_base_url()
EXPECT_CUDA_RUNTIME = os.environ.get("KFORGE_EXPECT_CUDA_RUNTIME", "").strip().lower() in {
    "1",
    "true",
    "yes",
}
print(f"[smoke] base_url={BASE_URL}", flush=True)


def request(method: str, path: str, payload=None, timeout: int = 20):
    url = BASE_URL + path
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        text = body.decode("utf-8", errors="replace")
        content_type = resp.headers.get("Content-Type", "")
        if "application/json" in content_type:
            return resp.status, json.loads(text)
        return resp.status, text


def walker(name: str, payload):
    status, body = request("POST", f"/walker/{name}", payload)
    assert status == 200, (name, status)
    assert body.get("ok", False), (name, body)
    reports = body.get("data", {}).get("reports", [])
    return reports[0] if reports else None


for route in ["/", "/project/desktop-smoke-route"]:
    status, html = request("GET", route)
    assert status == 200, (route, status)
    assert '<div id="root"></div>' in html, route
print("[smoke] ui routes ok", flush=True)

cfg0 = walker("GetConfig", {})
assert isinstance(cfg0, dict) and "api_keys" in cfg0, cfg0
marker = f"desktop-smoke-{int(time.time())}"
cfg1 = json.loads(json.dumps(cfg0))
cfg1["api_keys"]["openai"] = marker
config_path = Path.home() / ".config/com.kernelforge.desktop/config.json"
assert walker("SaveConfig", {"cfg": cfg1}) is True
assert walker("GetConfig", {})["api_keys"]["openai"] == marker
saved_cfg = json.loads(config_path.read_text(encoding="utf-8"))
assert saved_cfg.get("api_keys", {}).get("openai") == marker, saved_cfg
assert walker("SaveConfig", {"cfg": cfg0}) is True
assert walker("GetConfig", {})["api_keys"]["openai"] == cfg0["api_keys"]["openai"]
restored_cfg = json.loads(config_path.read_text(encoding="utf-8"))
assert restored_cfg.get("api_keys", {}).get("openai") == cfg0["api_keys"]["openai"], restored_cfg
print("[smoke] config roundtrip ok", flush=True)

hw = walker("DetectHardware", {})
assert isinstance(hw, dict), hw
assert "preferred_target" in hw and "gpus" in hw, hw
print(
    json.dumps(
        {
            "smoke": "hardware",
            "gpu_present": hw.get("gpu_present"),
            "cuda_available": hw.get("cuda_available"),
            "preferred_target": hw.get("preferred_target"),
            "gpu0": hw.get("gpus", [None])[0],
        },
        indent=2,
    ),
    flush=True,
)
if EXPECT_CUDA_RUNTIME:
    assert hw.get("cuda_available") is True, hw
    gpu0 = hw.get("gpus", [None])[0] if hw.get("gpus") else None
    assert gpu0 and gpu0.get("runtime_usable") is True, hw
    assert hw.get("preferred_target") == "cuda", hw

system = walker("GetSystemInfo", {})
assert isinstance(system, dict), system
print("[smoke] system info ok", flush=True)

selected_backend = "cuda" if EXPECT_CUDA_RUNTIME else "cpu"

project_name = f"Desktop Smoke {int(time.time())}"
model_code = """\
import torch

class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)

def sample_inputs():
    return [torch.randn(1, 4)]
"""

created_name = None
try:
    create = walker(
        "CreateProject",
        {
            "projectName": project_name,
            "code": model_code,
            "selectedGpuInfo": {},
            "selectedBackend": selected_backend,
        },
    )
    assert create and create.get("success") is True, create
    created_name = create["name"]
    print(f"[smoke] created project {created_name}", flush=True)
    project_dir = Path.home() / ".local/share/com.kernelforge.desktop/projects" / created_name
    assert project_dir.exists(), project_dir
    assert (project_dir / "model.py").exists(), project_dir / "model.py"

    projects = walker("GetProjects", {})
    project_names = [p["name"] for p in projects.get("projects", [])]
    assert created_name in project_names, project_names

    final_status = None
    polls_used = 0
    for i in range(25):
        final_status = walker(
            "GetProjectStatus",
            {"projectName": created_name, "currentGpuInfo": {}},
        )
        profile = final_status.get("profile", {}) if isinstance(final_status, dict) else {}
        profile_status = str(profile.get("status", "")).lower()
        print(
            json.dumps(
                {
                    "smoke": "status_poll",
                    "poll": i,
                    "profile_status": profile_status,
                    "phase": profile.get("phase"),
                    "message": profile.get("message"),
                    "progress_percent": profile.get("progress_percent"),
                    "backend": final_status.get("backend") if isinstance(final_status, dict) else None,
                }
            ),
            flush=True,
        )
        polls_used = i + 1
        if profile_status not in {"running", "queued"}:
            break
        time.sleep(1)

    profile = final_status.get("profile", {}) if isinstance(final_status, dict) else {}
    assert str(profile.get("status", "")).lower() == "completed", final_status
    assert str(final_status.get("backend", "")).lower() == selected_backend, final_status
    assert polls_used <= 12, final_status

    log_payload = walker(
        "GetJobLog",
        {"projectName": created_name, "jobKey": "profile", "lines": 40},
    )
    assert log_payload.get("lines"), log_payload
    summary_path = project_dir / "io" / "summary.json"
    assert summary_path.exists(), summary_path
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert str(summary.get("device", "")).lower() == selected_backend, summary
    print("[smoke] project lifecycle ok", flush=True)
finally:
    if created_name:
        remove = walker("DeleteProject", {"projectName": created_name})
        assert remove and remove.get("success") is True, remove
        names = []
        for _ in range(6):
            remaining = walker("GetProjects", {})
            names = [p["name"] for p in remaining.get("projects", [])]
            if created_name not in names:
                break
            time.sleep(1)
        assert created_name not in names, names
        print(f"[smoke] deleted project {created_name}", flush=True)

print("[smoke] ok", flush=True)
PY
