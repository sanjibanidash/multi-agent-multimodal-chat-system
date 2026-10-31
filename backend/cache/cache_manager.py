"""
cache/cache_manager.py
-----------------------
Simple TTL-based in-memory cache to avoid repeating expensive LLM calls.

Difference from session memory:
  - CACHE  = stores (query → response) globally, shared across all users.
             Saves money & time when the same question is asked repeatedly.
  - MEMORY = per-session conversation thread. Gives the LLM context.

Cache key = hash of (agent_type + cleaned_query).
TTL       = configurable via CACHE_TTL_SECONDS env variable.
Max size  = configurable via CACHE_MAX_SIZE env variable.
"""
import hashlib
import time
from threading import Lock
from typing import Any, Optional

from backend.config.settings import settings


class _CacheEntry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, ttl: int):
        self.value = value
        self.expires_at = time.monotonic() + ttl


class CacheManager:
    def __init__(self, max_size: int = 200, ttl: int = 3600):
        self._store: dict[str, _CacheEntry] = {}
        self._max_size = max_size
        self._ttl = ttl
        self._lock = Lock()  # thread-safe for FastAPI's async workers

    # ── Key generation ───────────────────────────────────────────────────────

    @staticmethod
    def make_key(agent: str, query: str) -> str:
        """Create a deterministic cache key from the agent name and user query."""
        # Normalize whitespace before hashing to avoid trivial cache misses
        normalized = " ".join(query.lower().split())
        raw = f"{agent}::{normalized}"
        return hashlib.sha256(raw.encode()).hexdigest()

    # ── Public API ───────────────────────────────────────────────────────────

    def get(self, key: str) -> tuple[bool, Optional[Any]]:
        """
        Look up a key.

        Returns:
            (True, value)  on cache hit
            (False, None)  on cache miss or expired entry
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return False, None
            if time.monotonic() > entry.expires_at:
                del self._store[key]
                return False, None
            return True, entry.value

    def set(self, key: str, value: Any) -> None:
        """Store a value. Evicts oldest entry if max_size is reached."""
        with self._lock:
            if len(self._store) >= self._max_size:
                # Evict the first (oldest) key — dict insertion order is guaranteed in Python 3.7+
                oldest_key = next(iter(self._store))
                del self._store[oldest_key]
            self._store[key] = _CacheEntry(value, self._ttl)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


# Singleton — import this everywhere
cache = CacheManager(
    max_size=settings.CACHE_MAX_SIZE,
    ttl=settings.CACHE_TTL_SECONDS,
)
