"""Redis-backed cache for session metadata and repeated retrieval queries.

`RedisCache` is used whenever FINGUARD_REDIS_URL is set. `InMemoryCache` keeps
the same interface for tests and single-process demos.

Cache failures never fail a request: a Redis outage degrades to a cache miss.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class Cache(Protocol):
    def get_json(self, key: str) -> Any | None: ...
    def set_json(self, key: str, value: Any, ttl_s: int) -> None: ...
    def ping(self) -> bool: ...


def retrieval_key(query: str) -> str:
    normalised = " ".join(query.lower().split())
    return "fg:retrieval:" + hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def session_key(session_id: str) -> str:
    return f"fg:session:{session_id}"


class InMemoryCache:
    def __init__(self) -> None:
        self._data: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()

    def get_json(self, key: str) -> Any | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires, raw = item
            if expires < time.monotonic():
                del self._data[key]
                return None
            return json.loads(raw)

    def set_json(self, key: str, value: Any, ttl_s: int) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + ttl_s, json.dumps(value))

    def ping(self) -> bool:
        return True


class RedisCache:
    def __init__(self, url: str) -> None:
        import redis

        self._client = redis.Redis.from_url(url, socket_timeout=0.5, socket_connect_timeout=0.5)

    def get_json(self, key: str) -> Any | None:
        try:
            raw = self._client.get(key)
        except Exception as exc:  # cache outage = cache miss
            logger.warning("redis get failed: %s", exc)
            return None
        return json.loads(raw) if raw else None

    def set_json(self, key: str, value: Any, ttl_s: int) -> None:
        try:
            self._client.set(key, json.dumps(value), ex=ttl_s)
        except Exception as exc:
            logger.warning("redis set failed: %s", exc)

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:
            return False


def build_cache(redis_url: str) -> Cache:
    return RedisCache(redis_url) if redis_url else InMemoryCache()
