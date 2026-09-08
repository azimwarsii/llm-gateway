import httpx

from tests.conftest import MEDIUM, REMOTE, SMALL, completion, sse


def simple(text="What is the capital of France?", **kw):
    return {"model": "auto", "messages": [{"role": "user", "content": text}], "temperature": 0, **kw}


async def test_routes_simple_to_small_and_reports_metadata(client, mock):
    mock.post(f"{SMALL}/chat/completions").mock(return_value=httpx.Response(200, json=completion("Paris")))
    r = await client.post("/v1/chat/completions", json=simple())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Paris"
    g = body["gateway"]
    assert g["backend"] == "small-1" and g["tier"] == "small" and g["cache"] == "miss"
    assert g["cost_usd"] > 0


async def test_cache_hit_on_repeat(client, mock):
    route = mock.post(f"{SMALL}/chat/completions").mock(return_value=httpx.Response(200, json=completion("Paris")))
    await client.post("/v1/chat/completions", json=simple())
    r = await client.post("/v1/chat/completions", json=simple())
    assert r.json()["gateway"]["cache"] == "hit"
    assert route.call_count == 1


async def test_retry_then_fallback_to_next_tier(client, mock):
    mock.post(f"{SMALL}/chat/completions").mock(return_value=httpx.Response(503, text="overloaded"))
    mock.post(f"{MEDIUM}/chat/completions").mock(return_value=httpx.Response(200, json=completion("ok")))
    r = await client.post("/v1/chat/completions", json=simple())
    assert r.status_code == 200
    g = r.json()["gateway"]
    assert g["backend"] == "medium-1" and g["requested_tier"] == "small"
    outcomes = [a["outcome"] for a in g["attempts"]]
    assert outcomes.count("retryable_error") == 2  # 1 try + 1 retry
    assert outcomes[-1] == "ok"


async def test_circuit_opens_and_skips_backend(client, mock):
    mock.post(f"{SMALL}/chat/completions").mock(return_value=httpx.Response(500, text="boom"))
    mock.post(f"{MEDIUM}/chat/completions").mock(return_value=httpx.Response(200, json=completion("ok")))
    await client.post("/v1/chat/completions", json=simple("q1"))  # trips the breaker (threshold=2)
    r = await client.post("/v1/chat/completions", json=simple("q2"))
    first = r.json()["gateway"]["attempts"][0]
    assert first["backend"] == "small-1" and first["outcome"] == "skipped" and first["error"] == "circuit_open"
    st = client.app.state.engine.breakers.snapshot()
    assert st["small-1"]["state"] == "open"


async def test_all_backends_down_returns_503(client, mock):
    for base in (SMALL, MEDIUM, REMOTE):
        mock.post(f"{base}/chat/completions").mock(side_effect=httpx.ConnectError("down"))
    r = await client.post("/v1/chat/completions", json=simple())
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "no_backend_available"


async def test_fatal_4xx_does_not_retry_but_falls_back(client, mock):
    small = mock.post(f"{SMALL}/chat/completions").mock(return_value=httpx.Response(400, text="bad"))
    mock.post(f"{MEDIUM}/chat/completions").mock(return_value=httpx.Response(200, json=completion("ok")))
    r = await client.post("/v1/chat/completions", json=simple())
    assert r.status_code == 200 and small.call_count == 1


async def test_tenant_auth_and_tier_restriction(client, mock):
    mock.post(f"{SMALL}/chat/completions").mock(return_value=httpx.Response(200, json=completion("ok")))
    r = await client.post("/v1/chat/completions", json=simple(), headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401
    h = {"Authorization": "Bearer tiny-key"}
    r = await client.post("/v1/chat/completions", json=simple(), headers={**h, "X-Route-Tier": "remote"})
    assert r.status_code == 403


async def test_rate_limit_and_budget(client, mock):
    mock.post(f"{SMALL}/chat/completions").mock(return_value=httpx.Response(200, json=completion("ok", pt=20, ct=20)))
    h = {"Authorization": "Bearer tiny-key"}
    r1 = await client.post("/v1/chat/completions", json=simple("a"), headers=h)
    assert r1.status_code == 200
    r2 = await client.post("/v1/chat/completions", json=simple("b"), headers=h)
    # budget=30 tokens, first call used 40 -> second is rejected by budget (or rate limit at 2/min on third)
    assert r2.status_code == 429
    u = await client.get("/v1/usage", headers=h)
    assert u.json()["tokens_used"] == 40


async def test_streaming_passthrough(client, mock):
    mock.post(f"{SMALL}/chat/completions").mock(
        return_value=httpx.Response(200, text=sse(["Pa", "ris"]), headers={"content-type": "text/event-stream"})
    )
    async with client.stream("POST", "/v1/chat/completions", json=simple(stream=True)) as r:
        assert r.status_code == 200
        text = (await r.aread()).decode()
    assert '"content": "Pa"' in text and "[DONE]" in text


async def test_streaming_falls_back_before_first_byte(client, mock):
    mock.post(f"{SMALL}/chat/completions").mock(return_value=httpx.Response(503, text="x"))
    mock.post(f"{MEDIUM}/chat/completions").mock(
        return_value=httpx.Response(200, text=sse(["ok"]), headers={"content-type": "text/event-stream"})
    )
    async with client.stream("POST", "/v1/chat/completions", json=simple(stream=True)) as r:
        assert r.status_code == 200
        assert "ok" in (await r.aread()).decode()


async def test_health_models_metrics(client, mock):
    r = await client.get("/health")
    assert r.json()["status"] == "ok"
    r = await client.get("/v1/models")
    assert "auto" in [m["id"] for m in r.json()["data"]]
    r = await client.get("/metrics")
    assert "gateway_requests_total" in r.text


async def test_cache_bypass_header(client, mock):
    route = mock.post(f"{SMALL}/chat/completions").mock(return_value=httpx.Response(200, json=completion("Paris")))
    await client.post("/v1/chat/completions", json=simple())
    r = await client.post("/v1/chat/completions", json=simple(), headers={"X-Cache": "bypass"})
    assert r.json()["gateway"]["cache"] == "bypass" and route.call_count == 2
