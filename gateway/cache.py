"""Response cache keyed on a hash of the normalised request. Redis-backed with in-memory fallback."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from .config import CacheConfig


def cache_key(body: dict[str, Any], tier: str) -> str:
    # Only deterministic requests are cacheable.
    norm = {
        "messages": body.get("messages"),
        "temperature": body.get("temperature", 1.0),
        "max_tokens": body.get("max_tokens"),
        "tools": body.get("tools"),
        "response_format": body.get("response_format"),
        "tier": tier,
    }
    return "resp:" + hashlib.sha256(json.dumps(norm, sort_keys=True).encode()).hexdigest()


def cacheable(body: dict[str, Any]) -> bool:
    return not body.get("stream") and (body.get("temperature") in (None, 0, 0.0))


class ResponseCache:
    def __init__(self, cfg: CacheConfig, redis_client=None):
        self.cfg = cfg
        self.r = redis_client
        self._mem: dict[str, tuple[float, str]] = {}
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> dict[str, Any] | None:
        if not self.cfg.enabled:
            return None
        raw: str | None = None
        if self.r is not None:
            try:
                raw = await self.r.get(key)
            except Exception:
                raw = None
        if raw is None:
            item = self._mem.get(key)
            if item and item[0] > time.time():
                raw = item[1]
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(raw)

    async def set(self, key: str, value: dict[str, Any]) -> None:
        if not self.cfg.enabled:
            return
        raw = json.dumps(value)
        if self.r is not None:
            try:
                await self.r.set(key, raw, ex=self.cfg.ttl_s)
                return
            except Exception:
                pass
        self._mem[key] = (time.time() + self.cfg.ttl_s, raw)
