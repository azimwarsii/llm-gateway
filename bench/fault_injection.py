"""Chaos test: fires N requests while the small backend is repeatedly killed and restarted.

Reports the success rate the gateway maintained and how many requests fell back.
Requires docker compose running with the `small` profile.

  python bench/fault_injection.py --requests 100 --kill-every 15
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time

import httpx


async def chaos(service: str, every: float, stop: asyncio.Event) -> None:
    while not stop.is_set():
        await asyncio.sleep(every)
        subprocess.run(["docker", "compose", "kill", service], check=False, capture_output=True)
        print(f"  [chaos] killed {service}")
        await asyncio.sleep(3)
        subprocess.run(["docker", "compose", "--profile", "small", "start", service], check=False, capture_output=True)
        print(f"  [chaos] restarted {service}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://localhost:8080")
    ap.add_argument("--api-key", default="bench-key")
    ap.add_argument("--requests", type=int, default=100)
    ap.add_argument("--kill-every", type=float, default=15.0)
    ap.add_argument("--service", default="vllm-small")
    a = ap.parse_args()
    stop = asyncio.Event()
    task = asyncio.create_task(chaos(a.service, a.kill_every, stop))
    ok = fb = 0
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=120) as c:
        for i in range(a.requests):
            body = {
                "model": "auto",
                "messages": [{"role": "user", "content": f"Say the number {i} in words."}],
                "max_tokens": 16,
            }
            try:
                r = await c.post(
                    f"{a.gateway}/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {a.api_key}"}
                )
                if r.status_code == 200:
                    ok += 1
                    g = r.json()["gateway"]
                    if g["tier"] != g["requested_tier"]:
                        fb += 1
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
    stop.set()
    task.cancel()
    print(
        json.dumps(
            {
                "requests": a.requests,
                "ok": ok,
                "success_rate": ok / a.requests,
                "fell_back": fb,
                "elapsed_s": round(time.perf_counter() - t0, 1),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
