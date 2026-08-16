from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

Tier = Literal["small", "medium", "remote"]
TIER_ORDER: list[Tier] = ["small", "medium", "remote"]


class BackendConfig(BaseModel):
    name: str
    tier: Tier
    base_url: str
    model: str
    api_key_env: str | None = None
    price_input_per_m: float = 0.0
    price_output_per_m: float = 0.0
    max_context: int = 4096
    timeout_s: float = 60.0

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


class RouterConfig(BaseModel):
    small_max: float = 0.35
    medium_max: float = 0.70
    allow_tier_override: bool = True


class ResilienceConfig(BaseModel):
    max_retries: int = 2
    backoff_base_s: float = 0.5
    backoff_max_s: float = 4.0
    circuit_failure_threshold: int = 3
    circuit_recovery_s: float = 30.0
    health_check_interval_s: float = 10.0


class CacheConfig(BaseModel):
    enabled: bool = True
    ttl_s: int = 3600


class TenantConfig(BaseModel):
    api_key: str
    daily_token_budget: int = 1_000_000
    requests_per_minute: int = 60
    allowed_tiers: list[Tier] = Field(default_factory=lambda: list(TIER_ORDER))


class TenantsConfig(BaseModel):
    default: str | None = None
    keys: dict[str, TenantConfig] = Field(default_factory=dict)


class GatewayConfig(BaseModel):
    backends: list[BackendConfig]
    router: RouterConfig = RouterConfig()
    resilience: ResilienceConfig = ResilienceConfig()
    cache: CacheConfig = CacheConfig()
    tenants: TenantsConfig = TenantsConfig()

    def backends_for_tier(self, tier: Tier) -> list[BackendConfig]:
        return [b for b in self.backends if b.tier == tier]

    def backend(self, name: str) -> BackendConfig | None:
        return next((b for b in self.backends if b.name == name), None)


def load_config(path: str | os.PathLike | None = None) -> GatewayConfig:
    path = Path(path or os.environ.get("GATEWAY_CONFIG", "config/gateway.yaml"))
    with open(path) as f:
        raw = yaml.safe_load(f)
    return GatewayConfig.model_validate(raw)
