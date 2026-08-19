"""Retries with exponential backoff and a per-backend circuit breaker."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum

from .config import ResilienceConfig


class CircuitState(str, Enum):
    CLOSED = "closed"  # healthy, traffic flows
    OPEN = "open"  # failing, traffic blocked
    HALF_OPEN = "half_open"  # probing with one request


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 3
    recovery_s: float = 30.0
    failures: int = 0
    state: CircuitState = CircuitState.CLOSED
    opened_at: float = 0.0
    last_error: str = ""
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def allow(self) -> bool:
        if self.state is CircuitState.CLOSED:
            return True
        if self.state is CircuitState.OPEN and time.monotonic() - self.opened_at >= self.recovery_s:
            self.state = CircuitState.HALF_OPEN
            return True
        return self.state is CircuitState.HALF_OPEN

    async def record_success(self) -> None:
        async with self._lock:
            self.failures = 0
            self.state = CircuitState.CLOSED

    async def record_failure(self, err: str) -> None:
        async with self._lock:
            self.failures += 1
            self.last_error = err[:200]
            if self.state is CircuitState.HALF_OPEN or self.failures >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.opened_at = time.monotonic()

    def snapshot(self) -> dict:
        return {"state": self.state.value, "failures": self.failures, "last_error": self.last_error}


class Breakers:
    def __init__(self, cfg: ResilienceConfig):
        self.cfg = cfg
        self._b: dict[str, CircuitBreaker] = {}

    def get(self, name: str) -> CircuitBreaker:
        if name not in self._b:
            self._b[name] = CircuitBreaker(name, self.cfg.circuit_failure_threshold, self.cfg.circuit_recovery_s)
        return self._b[name]

    def snapshot(self) -> dict[str, dict]:
        return {k: v.snapshot() for k, v in self._b.items()}


def backoff_delay(attempt: int, cfg: ResilienceConfig) -> float:
    """Full-jitter exponential backoff."""
    cap = min(cfg.backoff_max_s, cfg.backoff_base_s * (2**attempt))
    return random.uniform(0, cap)


class RetryableError(Exception):
    """Transient failure: timeouts, 5xx, 429, connection errors."""


class FatalError(Exception):
    """Non-retryable failure: 4xx other than 429, malformed responses."""
