from __future__ import annotations

from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
LOCAL_DIR = "/home/gb10/model-cache/Qwen3.5-35B-A3B"


def main() -> int:
    path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        max_workers=8,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
