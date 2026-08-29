"""Test fixtures: a gateway wired to fake OpenAI-compatible backends via respx."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from httpx import ASGITransport, AsyncClient

from gateway.backends import OpenAICompatBackend
from gateway.config import (
    BackendConfig,
    CacheConfig,
    GatewayConfig,
    ResilienceConfig,
    RouterConfig,
    TenantConfig,
    TenantsConfig,
)
from gateway.main import build_app

SMALL = "http://small.test/v1"
MEDIUM = "http://medium.test/v1"
REMOTE = "http://remote.test/v1"


def make_cfg(**over) -> GatewayConfig:
    cfg = GatewayConfig(
        backends=[
            BackendConfig(
                name="small-1",
                tier="small",
                base_url=SMALL,
                model="m-small",
                price_input_per_m=0.05,
                price_output_per_m=0.2,
            ),
            BackendConfig(
                name="medium-1",
                tier="medium",
                base_url=MEDIUM,
                model="m-medium",
                price_input_per_m=0.1,
                price_output_per_m=0.5,
            ),
            BackendConfig(
                name="remote-1",
                tier="remote",
                base_url=REMOTE,
                model="m-remote",
                price_input_per_m=0.6,
                price_output_per_m=0.8,
            ),
        ],
        router=RouterConfig(),
        resilience=ResilienceConfig(
            max_retries=1,
            backoff_base_s=0.0,
            backoff_max_s=0.0,
            circuit_failure_threshold=2,
            circuit_recovery_s=60,
            health_check_interval_s=1000,
        ),
        cache=CacheConfig(enabled=True, ttl_s=60),
        tenants=TenantsConfig(
            default="dev",
            keys={
                "dev": TenantConfig(api_key="dev-key", daily_token_budget=10_000, requests_per_minute=1000),
                "tiny": TenantConfig(
                    api_key="tiny-key", daily_token_budget=30, requests_per_minute=2, allowed_tiers=["small"]
                ),
            },
        ),
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def completion(text="hello", model="m", pt=10, ct=5):
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
    }


def sse(chunks):
    body = ""
    for c in chunks:
        body += "data: " + json.dumps({"choices": [{"delta": {"content": c}}]}) + "\n\n"
    return body + "data: [DONE]\n\n"


@pytest.fixture
def mock():
    with respx.mock(assert_all_called=False) as m:
        for base in (SMALL, MEDIUM, REMOTE):
            m.get(f"{base}/models").mock(return_value=httpx.Response(200, json={"data": []}))
        yield m


@pytest.fixture
async def client(mock):
    cfg = make_cfg()
    shared = httpx.AsyncClient(timeout=5)  # respx patches this transport
    backends = {b.name: OpenAICompatBackend(b, client=shared) for b in cfg.backends}
    app = build_app(cfg, backends)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://gw") as c:
            c.app = app  # type: ignore[attr-defined]
            yield c
    await shared.aclose()
