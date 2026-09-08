"""Ties routing, resilience, caching, tenants and metrics together for one request."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from . import metrics as M
from .backends import OpenAICompatBackend
from .cache import ResponseCache, cache_key, cacheable
from .config import TIER_ORDER, BackendConfig, GatewayConfig, Tier
from .resilience import Breakers, FatalError, RetryableError, backoff_delay
from .router import RouteDecision, route
from .schemas import ChatCompletionRequest
from .tenants import Tenant, TenantManager
from .tokens import count_tokens

log = logging.getLogger("gateway")


@dataclass
class Attempt:
    backend: str
    tier: str
    outcome: str
    error: str = ""
    latency_ms: float = 0.0


class NoBackendAvailable(Exception):
    def __init__(self, attempts: list[Attempt]):
        super().__init__("all backends failed or unavailable")
        self.attempts = attempts


class Engine:
    def __init__(
        self, cfg: GatewayConfig, backends: dict[str, OpenAICompatBackend], cache: ResponseCache, tenants: TenantManager
    ):
        self.cfg = cfg
        self.backends = backends
        self.cache = cache
        self.tenants = tenants
        self.breakers = Breakers(cfg.resilience)
        self.healthy: dict[str, bool] = {b: True for b in backends}
        self._health_task: asyncio.Task | None = None

    # ---------- health ----------
    async def start(self) -> None:
        self._health_task = asyncio.create_task(self._health_loop())

    async def stop(self) -> None:
        if self._health_task:
            self._health_task.cancel()
        for b in self.backends.values():
            await b.aclose()

    async def _health_loop(self) -> None:
        while True:
            await self.check_health()
            await asyncio.sleep(self.cfg.resilience.health_check_interval_s)

    async def check_health(self) -> dict[str, bool]:
        results = await asyncio.gather(*(b.health() for b in self.backends.values()), return_exceptions=True)
        for name, ok in zip(self.backends, results):
            up = ok is True
            self.healthy[name] = up
            M.BACKEND_UP.labels(name).set(1 if up else 0)
            M.CIRCUIT.labels(name).set(1 if self.breakers.get(name).state.value == "open" else 0)
        return dict(self.healthy)

    # ---------- candidate selection ----------
    def candidates(self, tier: Tier, allowed: list[Tier]) -> list[tuple[Tier, BackendConfig]]:
        """Backends for the chosen tier first, then each stronger tier as fallback."""
        start = TIER_ORDER.index(tier)
        out: list[tuple[Tier, BackendConfig]] = []
        for t in TIER_ORDER[start:]:
            if t not in allowed:
                continue
            for b in self.cfg.backends_for_tier(t):
                out.append((t, b))
        return out

    def _usable(self, b: BackendConfig) -> tuple[bool, str]:
        if not self.healthy.get(b.name, True):
            return False, "unhealthy"
        if not self.breakers.get(b.name).allow():
            return False, "circuit_open"
        return True, ""

    # ---------- cost ----------
    @staticmethod
    def cost_usd(b: BackendConfig, usage: dict[str, Any]) -> float:
        pi = usage.get("prompt_tokens", 0) or 0
        co = usage.get("completion_tokens", 0) or 0
        return pi / 1e6 * b.price_input_per_m + co / 1e6 * b.price_output_per_m

    # ---------- main path ----------
    async def complete(
        self,
        req: ChatCompletionRequest,
        tenant: Tenant,
        tier_override: str | None = None,
        cache_bypass: bool = False,
    ) -> dict[str, Any]:
        decision: RouteDecision = route(req, self.cfg.router, tier_override)
        if decision.score >= 0:
            M.ROUTE_SCORE.observe(decision.score)
        await self.tenants.check_request(tenant, decision.tier)
        body = req.model_dump(exclude_none=True)

        key = cache_key(body, decision.tier) if (cacheable(body) and not cache_bypass) else None
        if key:
            hit = await self.cache.get(key)
            if hit:
                M.CACHE.labels("hit").inc()
                hit = dict(hit)
                hit["gateway"] = {**hit.get("gateway", {}), "cache": "hit"}
                return hit
            M.CACHE.labels("miss").inc()

        attempts: list[Attempt] = []
        chosen_tier = decision.tier
        for tier, bcfg in self.candidates(decision.tier, tenant.cfg.allowed_tiers):
            ok, why = self._usable(bcfg)
            if not ok:
                attempts.append(Attempt(bcfg.name, tier, "skipped", why))
                continue
            if tier != chosen_tier:
                M.FALLBACKS.labels(chosen_tier, tier).inc()
                chosen_tier = tier
            backend = self.backends[bcfg.name]
            breaker = self.breakers.get(bcfg.name)
            for attempt in range(self.cfg.resilience.max_retries + 1):
                t0 = time.perf_counter()
                try:
                    raw = await backend.chat(body)
                    lat = time.perf_counter() - t0
                    await breaker.record_success()
                    usage = raw.get("usage") or self._estimate_usage(body, raw)
                    cost = self.cost_usd(bcfg, usage)
                    M.REQUESTS.labels(bcfg.name, tier, "ok").inc()
                    M.LATENCY.labels(bcfg.name, tier).observe(lat)
                    M.TOKENS.labels(bcfg.name, "input").inc(usage.get("prompt_tokens", 0) or 0)
                    M.TOKENS.labels(bcfg.name, "output").inc(usage.get("completion_tokens", 0) or 0)
                    M.COST.labels(bcfg.name, tenant.name).inc(cost)
                    await self.tenants.record_usage(tenant, usage.get("total_tokens", 0) or 0)
                    attempts.append(Attempt(bcfg.name, tier, "ok", latency_ms=round(lat * 1000, 1)))
                    raw["model"] = f"{bcfg.name}"
                    raw["gateway"] = {
                        "backend": bcfg.name,
                        "tier": tier,
                        "requested_tier": decision.tier,
                        "route_score": decision.score,
                        "route_reason": decision.reason,
                        "route_features": decision.features,
                        "latency_ms": round(lat * 1000, 1),
                        "cost_usd": round(cost, 8),
                        "cache": "miss" if key else "bypass",
                        "attempts": [a.__dict__ for a in attempts],
                    }
                    if key:
                        await self.cache.set(key, raw)
                    return raw
                except RetryableError as e:
                    lat = time.perf_counter() - t0
                    await breaker.record_failure(str(e))
                    M.REQUESTS.labels(bcfg.name, tier, "retryable_error").inc()
                    attempts.append(Attempt(bcfg.name, tier, "retryable_error", str(e)[:120], round(lat * 1000, 1)))
                    log.warning("backend %s attempt %d failed: %s", bcfg.name, attempt, e)
                    if attempt < self.cfg.resilience.max_retries and breaker.allow():
                        await asyncio.sleep(backoff_delay(attempt, self.cfg.resilience))
                        continue
                    break  # move to next candidate
                except FatalError as e:
                    M.REQUESTS.labels(bcfg.name, tier, "fatal_error").inc()
                    attempts.append(Attempt(bcfg.name, tier, "fatal_error", str(e)[:120]))
                    log.error("backend %s fatal: %s", bcfg.name, e)
                    break
        raise NoBackendAvailable(attempts)

    async def stream(
        self, req: ChatCompletionRequest, tenant: Tenant, tier_override: str | None = None
    ) -> AsyncIterator[bytes]:
        """Streaming path: fallback only before the first byte is sent; after that, errors propagate."""
        decision = route(req, self.cfg.router, tier_override)
        await self.tenants.check_request(tenant, decision.tier)
        body = req.model_dump(exclude_none=True)
        attempts: list[Attempt] = []
        for tier, bcfg in self.candidates(decision.tier, tenant.cfg.allowed_tiers):
            ok, why = self._usable(bcfg)
            if not ok:
                attempts.append(Attempt(bcfg.name, tier, "skipped", why))
                continue
            backend = self.backends[bcfg.name]
            breaker = self.breakers.get(bcfg.name)
            t0 = time.perf_counter()
            first = True
            try:
                async for chunk in backend.chat_stream(body):
                    if first:
                        M.TTFT.labels(bcfg.name).observe(time.perf_counter() - t0)
                        first = False
                    yield chunk
                await breaker.record_success()
                M.REQUESTS.labels(bcfg.name, tier, "ok").inc()
                M.LATENCY.labels(bcfg.name, tier).observe(time.perf_counter() - t0)
                # Streaming responses do not carry usage; estimate from the prompt.
                await self.tenants.record_usage(tenant, count_tokens(req.prompt_text()) + (req.max_tokens or 256))
                return
            except RetryableError as e:
                await breaker.record_failure(str(e))
                M.REQUESTS.labels(bcfg.name, tier, "retryable_error").inc()
                attempts.append(Attempt(bcfg.name, tier, "retryable_error", str(e)[:120]))
                if not first:
                    raise  # already streamed bytes; cannot fall back transparently
            except FatalError as e:
                M.REQUESTS.labels(bcfg.name, tier, "fatal_error").inc()
                attempts.append(Attempt(bcfg.name, tier, "fatal_error", str(e)[:120]))
                if not first:
                    raise
        raise NoBackendAvailable(attempts)

    @staticmethod
    def _estimate_usage(body: dict[str, Any], raw: dict[str, Any]) -> dict[str, int]:
        prompt = "\n".join(
            m.get("content") or "" for m in body.get("messages", []) if isinstance(m.get("content"), str)
        )
        out = ""
        try:
            out = raw["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            pass
        p, c = count_tokens(prompt), count_tokens(out)
        return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}
