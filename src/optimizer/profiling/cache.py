"""Small in-memory cache for profiler responses."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple


class ProfileCache:
    def __init__(self) -> None:
        self._store: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    def get(self, key: str, ttl_seconds: float) -> Optional[Dict[str, Any]]:
        entry = self._store.get(key)
        if not entry:
            return None
        ts, payload = entry
        if (time.time() - ts) > ttl_seconds:
            return None
        return payload

    def set(self, key: str, payload: Dict[str, Any]) -> None:
        self._store[key] = (time.time(), payload)


_CACHE = ProfileCache()


def get_cache() -> ProfileCache:
    return _CACHE

