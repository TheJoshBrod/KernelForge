#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python3 -u - <<'PY'
import os
import re
import subprocess
import sys
import time
import urllib.request


def proc_text() -> str:
    return subprocess.check_output(
        [
            "bash",
            "-lc",
            "pgrep -af 'kernel-forge-desktop|kf-ui-server.py|WebKitWebProcess' || true",
        ],
        text=True,
    )


def detect() -> tuple[str, str]:
    ui_pattern = re.compile(r"kf-ui-server\.py --bind 127\.0\.0\.1 --port (\d+)")
    webkit_pattern = re.compile(r"^(\d+)\s+.*WebKitWebProcess", re.MULTILINE)
    for _ in range(20):
        text = proc_text()
        ui_match = ui_pattern.search(text)
        webkit_match = webkit_pattern.search(text)
        if ui_match and webkit_match:
            return f"http://127.0.0.1:{ui_match.group(1)}", webkit_match.group(1)
        time.sleep(0.5)
    raise SystemExit("Could not find a running desktop UI server and WebKitWebProcess.")


def request(path: str) -> tuple[int, str, float]:
    base, _ = detect()
    started = time.perf_counter()
    with urllib.request.urlopen(base + path, timeout=10) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return resp.status, body, elapsed_ms


def sample_webkit_cpu(pid: str) -> float:
    out = subprocess.check_output(
        ["bash", "-lc", f"pidstat -h -u -p {pid} 1 5"],
        text=True,
    )
    values = []
    for line in out.splitlines():
        if "WebKitWebProces" not in line:
            continue
        cols = line.split()
        if len(cols) < 8:
            continue
        try:
            values.append(float(cols[7]))
        except ValueError:
            pass
    if not values:
        raise SystemExit("Failed to parse WebKit CPU samples.")
    return sum(values) / len(values)


base_url, webkit_pid = detect()
print(f"[perf] base_url={base_url}")
print(f"[perf] webkit_pid={webkit_pid}")

for route in ["/", "/settings", "/new-project", "/project/perf-check-route"]:
    status, html, elapsed_ms = request(route)
    assert status == 200, (route, status)
    assert '<div id="root"></div>' in html, route
    print(f"[perf] route={route} ms={elapsed_ms:.1f}")
    assert elapsed_ms < 500.0, (route, elapsed_ms)

avg_cpu = sample_webkit_cpu(webkit_pid)
print(f"[perf] webkit_avg_cpu={avg_cpu:.2f}")
assert avg_cpu < 25.0, avg_cpu

print("[perf] ok")
PY
