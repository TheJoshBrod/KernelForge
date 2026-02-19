from __future__ import annotations

import errno
import os
import time
from contextlib import contextmanager
from pathlib import Path

import fcntl


@contextmanager
def file_lock(lock_path: Path, timeout_s: float = 20.0, poll_s: float = 0.05):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    start = time.time()
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as e:
                if e.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if (time.time() - start) >= timeout_s:
                    raise TimeoutError(f"Timed out acquiring lock {lock_path}")
                time.sleep(poll_s)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            os.close(fd)
        except Exception:
            pass

