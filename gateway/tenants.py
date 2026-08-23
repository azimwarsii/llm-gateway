"""Per-tenant API keys, daily token budgets and per-minute rate limits.

Counters live in Redis (atomic INCRBY with TTL) so multiple gateway replicas share them;
falls back to process memory when Redis is unavailable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .config import TenantConfig, TenantsConfig, Tier


class TenantError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class Tenant:
    name: str
    cfg: TenantConfig


class TenantManager:
    def __init__(self, cfg: TenantsConfig, redis_client=None):
        self.cfg = cfg
        self.r = redis_client
        self._by_key = {t.api_key: Tenant(n, t) for n, t in cfg.keys.items()}
        self._mem: dict[str, tuple[float, int]] = {}

    def authenticate(self, authorization: str | None) -> Tenant:
        key = None
        if authorization and authorization.lower().startswith("bearer "):
            key = authorization[7:].strip()
        if key and key in self._by_key:
            return self._by_key[key]
        if key is None and self.cfg.default and self.cfg.default in self.cfg.keys:
            return Tenant(self.cfg.default, self.cfg.keys[self.cfg.default])
        raise TenantError(401, "invalid or missing API key")

    async def _incr(self, key: str, amount: int, ttl: int) -> int:
        if self.r is not None:
            try:
                pipe = self.r.pipeline()
                pipe.incrby(key, amount)
                pipe.expire(key, ttl, nx=True)
                res = await pipe.execute()
                return int(res[0])
            except Exception:
                pass
        exp, val = self._mem.get(key, (0.0, 0))
        if exp < time.time():
            exp, val = time.time() + ttl, 0
        val += amount
        self._mem[key] = (exp, val)
        return val

    async def _peek(self, key: str) -> int:
        if self.r is not None:
            try:
                v = await self.r.get(key)
                return int(v) if v else 0
            except Exception:
                pass
        exp, val = self._mem.get(key, (0.0, 0))
        return val if exp >= time.time() else 0

    @staticmethod
    def _day() -> str:
        return time.strftime("%Y%m%d", time.gmtime())

    async def check_request(self, t: Tenant, tier: Tier) -> None:
        if tier not in t.cfg.allowed_tiers:
            raise TenantError(403, f"tenant '{t.name}' may not use tier '{tier}'")
        minute = int(time.time() // 60)
        rpm = await self._incr(f"rl:{t.name}:{minute}", 1, 120)
        if rpm > t.cfg.requests_per_minute:
            raise TenantError(429, f"rate limit exceeded ({t.cfg.requests_per_minute}/min)")
        used = await self._peek(f"budget:{t.name}:{self._day()}")
        if used >= t.cfg.daily_token_budget:
            raise TenantError(429, f"daily token budget exhausted ({t.cfg.daily_token_budget})")

    async def record_usage(self, t: Tenant, tokens: int) -> int:
        return await self._incr(f"budget:{t.name}:{self._day()}", tokens, 60 * 60 * 48)

    async def usage(self, t: Tenant) -> dict:
        used = await self._peek(f"budget:{t.name}:{self._day()}")
        return {
            "tenant": t.name,
            "day": self._day(),
            "tokens_used": used,
            "daily_token_budget": t.cfg.daily_token_budget,
            "remaining": max(0, t.cfg.daily_token_budget - used),
        }
