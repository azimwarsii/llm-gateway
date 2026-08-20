"""Client for any OpenAI-compatible chat endpoint (vLLM, Ollama, Groq, OpenRouter, ...)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..config import BackendConfig
from ..resilience import FatalError, RetryableError


class OpenAICompatBackend:
    def __init__(self, cfg: BackendConfig, client: httpx.AsyncClient | None = None):
        self.cfg = cfg
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(cfg.timeout_s, connect=5.0))
        self._owns_client = client is None

    @property
    def name(self) -> str:
        return self.cfg.name

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            h["Authorization"] = f"Bearer {self.cfg.api_key}"
        return h

    def _payload(self, body: dict[str, Any]) -> dict[str, Any]:
        p = dict(body)
        p["model"] = self.cfg.model
        return p

    @staticmethod
    def _raise_for(resp: httpx.Response) -> None:
        if resp.status_code == 429 or resp.status_code >= 500:
            raise RetryableError(f"{resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise FatalError(f"{resp.status_code}: {resp.text[:200]}")

    async def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self._client.post(
                f"{self.cfg.base_url}/chat/completions", json=self._payload(body), headers=self._headers()
            )
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            raise RetryableError(f"{type(e).__name__}: {e}") from e
        self._raise_for(resp)
        try:
            return resp.json()
        except json.JSONDecodeError as e:
            raise FatalError("backend returned non-JSON") from e

    async def chat_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        """Yields raw SSE lines from the backend. The gateway re-emits them unchanged."""
        payload = self._payload(body)
        payload["stream"] = True
        try:
            async with self._client.stream(
                "POST", f"{self.cfg.base_url}/chat/completions", json=payload, headers=self._headers()
            ) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    self._raise_for(resp)
                async for line in resp.aiter_lines():
                    if line:
                        yield (line + "\n\n").encode()
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            raise RetryableError(f"{type(e).__name__}: {e}") from e

    async def health(self) -> bool:
        try:
            resp = await self._client.get(f"{self.cfg.base_url}/models", headers=self._headers(), timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
