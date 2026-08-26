from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .backends import OpenAICompatBackend
from .cache import ResponseCache
from .config import GatewayConfig, load_config
from .engine import Engine, NoBackendAvailable
from .schemas import ChatCompletionRequest, ModelInfo, ModelList
from .tenants import Tenant, TenantError, TenantManager

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gateway")


async def _redis():
    url = os.environ.get("REDIS_URL")
    if not url:
        return None
    try:
        import redis.asyncio as redis

        r = redis.from_url(url, decode_responses=True)
        await r.ping()
        log.info("connected to redis at %s", url)
        return r
    except Exception as e:  # pragma: no cover
        log.warning("redis unavailable (%s); using in-memory cache and counters", e)
        return None


def build_app(cfg: GatewayConfig | None = None, backends: dict[str, OpenAICompatBackend] | None = None) -> FastAPI:
    cfg = cfg or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        r = await _redis()
        bk = backends or {b.name: OpenAICompatBackend(b) for b in cfg.backends}
        engine = Engine(cfg, bk, ResponseCache(cfg.cache, r), TenantManager(cfg.tenants, r))
        app.state.engine = engine
        app.state.cfg = cfg
        await engine.check_health()
        await engine.start()
        log.info("gateway up with backends: %s", list(bk))
        yield
        await engine.stop()
        if r is not None:
            await r.aclose()

    app = FastAPI(title="llm-gateway", version="0.1.0", lifespan=lifespan)

    def tenant(request: Request, authorization: str | None = Header(default=None)) -> Tenant:
        try:
            return request.app.state.engine.tenants.authenticate(authorization)
        except TenantError as e:
            raise HTTPException(e.status, e.message)

    def admin(x_admin_key: str | None = Header(default=None)) -> None:
        expected = os.environ.get("ADMIN_API_KEY")
        if expected and x_admin_key != expected:
            raise HTTPException(401, "admin key required")

    @app.exception_handler(TenantError)
    async def _tenant_err(_: Request, e: TenantError):
        return JSONResponse({"error": {"message": e.message, "type": "tenant_error"}}, status_code=e.status)

    @app.exception_handler(NoBackendAvailable)
    async def _no_backend(_: Request, e: NoBackendAvailable):
        return JSONResponse(
            {
                "error": {
                    "message": str(e),
                    "type": "no_backend_available",
                    "attempts": [a.__dict__ for a in e.attempts],
                }
            },
            status_code=503,
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: ChatCompletionRequest,
        request: Request,
        t: Tenant = Depends(tenant),
        x_route_tier: str | None = Header(default=None),
    ):
        engine: Engine = request.app.state.engine
        if req.stream:
            gen = engine.stream(req, t, x_route_tier)
            # Pull the first chunk eagerly so routing/tenant errors surface as proper HTTP errors.
            first = await gen.__anext__()

            async def body():
                yield first
                async for chunk in gen:
                    yield chunk

            return StreamingResponse(body(), media_type="text/event-stream")
        return await engine.complete(req, t, x_route_tier)

    @app.get("/v1/models")
    async def models(request: Request, t: Tenant = Depends(tenant)):
        cfg: GatewayConfig = request.app.state.cfg
        ids = ["auto", "small", "medium", "remote"] + [b.name for b in cfg.backends]
        return ModelList(data=[ModelInfo(id=i) for i in ids])

    @app.get("/v1/usage")
    async def usage(request: Request, t: Tenant = Depends(tenant)):
        return await request.app.state.engine.tenants.usage(t)

    @app.get("/health")
    async def health(request: Request):
        engine: Engine = request.app.state.engine
        return {"status": "ok" if any(engine.healthy.values()) else "degraded", "backends": engine.healthy}

    @app.get("/admin/status", dependencies=[Depends(admin)])
    async def status(request: Request):
        engine: Engine = request.app.state.engine
        return {
            "backends": engine.healthy,
            "circuits": engine.breakers.snapshot(),
            "cache": {"hits": engine.cache.hits, "misses": engine.cache.misses},
        }

    @app.post("/admin/health-check", dependencies=[Depends(admin)])
    async def health_check(request: Request):
        return await request.app.state.engine.check_health()

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def get_app() -> FastAPI:
    return build_app()


app = build_app() if os.path.exists(os.environ.get("GATEWAY_CONFIG", "config/gateway.yaml")) else None
