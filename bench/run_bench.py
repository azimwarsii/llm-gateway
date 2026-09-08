"""Runs the prompt set through the gateway in several modes and writes per-request results.

Modes (each is one pass over prompts.jsonl):
  small   force tier=small       (X-Route-Tier: small)
  medium  force tier=medium
  remote  force tier=remote      (the "frontier-only" baseline)
  auto    let the router decide

Usage:
  python bench/run_bench.py --gateway http://localhost:8080 --api-key bench-key --modes small remote auto
  python bench/run_bench.py --modes auto --concurrency 4 --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

HERE = Path(__file__).parent


async def one(
    client: httpx.AsyncClient, gw: str, key: str, mode: str, row: dict, sem: asyncio.Semaphore, max_tokens: int
) -> dict:
    headers = {"Authorization": f"Bearer {key}", "X-Cache": "bypass"}  # a benchmark must not hit the cache
    if mode != "auto":
        headers["X-Route-Tier"] = mode
    body = {
        "model": "auto",
        "messages": [{"role": "user", "content": row["prompt"]}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{gw}/v1/chat/completions", json=body, headers=headers, timeout=180)
            wall = time.perf_counter() - t0
            if r.status_code != 200:
                return {
                    **row,
                    "mode": mode,
                    "ok": False,
                    "status": r.status_code,
                    "error": r.text[:200],
                    "wall_s": wall,
                }
            j = r.json()
            g = j.get("gateway", {})
            u = j.get("usage", {})
            return {
                **row,
                "mode": mode,
                "ok": True,
                "status": 200,
                "wall_s": round(wall, 3),
                "backend": g.get("backend"),
                "tier": g.get("tier"),
                "requested_tier": g.get("requested_tier"),
                "route_score": g.get("route_score"),
                "cache": g.get("cache"),
                "cost_usd": g.get("cost_usd", 0.0),
                "prompt_tokens": u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens"),
                "fallbacks": sum(1 for a in g.get("attempts", []) if a.get("outcome") != "ok"),
                "answer": (j["choices"][0]["message"].get("content") or "")[:4000],
            }
        except httpx.HTTPError as e:
            return {
                **row,
                "mode": mode,
                "ok": False,
                "status": 0,
                "error": str(e)[:200],
                "wall_s": time.perf_counter() - t0,
            }


def summarize(results: list[dict]) -> dict:
    by_mode: dict[str, list[dict]] = {}
    for r in results:
        by_mode.setdefault(r["mode"], []).append(r)
    out = {}
    for mode, rs in by_mode.items():
        oks = [r for r in rs if r["ok"]]
        lat = sorted(r["wall_s"] for r in oks)
        p = lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))], 3) if lat else None  # noqa: E731
        cost = sum(r.get("cost_usd", 0) for r in oks)
        tiers = {}
        for r in oks:
            tiers[r.get("tier")] = tiers.get(r.get("tier"), 0) + 1
        out[mode] = {
            "n": len(rs),
            "ok": len(oks),
            "success_rate": round(len(oks) / len(rs), 3) if rs else 0,
            "p50_s": p(0.5),
            "p95_s": p(0.95),
            "mean_s": round(statistics.mean(lat), 3) if lat else None,
            "total_cost_usd": round(cost, 6),
            "cost_per_1k_req_usd": round(cost / len(oks) * 1000, 4) if oks else None,
            "completion_tokens": sum(r.get("completion_tokens") or 0 for r in oks),
            "tier_mix": tiers,
            "retries_and_fallbacks": sum(r.get("fallbacks", 0) for r in oks),
        }
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://localhost:8080")
    ap.add_argument("--api-key", default="bench-key")
    ap.add_argument("--modes", nargs="+", default=["small", "remote", "auto"])
    ap.add_argument("--prompts", default=str(HERE / "prompts.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()

    rows = [json.loads(line) for line in Path(a.prompts).read_text().splitlines() if line.strip()]
    if a.limit:
        rows = rows[: a.limit]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    sem = asyncio.Semaphore(a.concurrency)
    results: list[dict] = []
    async with httpx.AsyncClient() as client:
        for mode in a.modes:
            t0 = time.perf_counter()
            res = await asyncio.gather(*(one(client, a.gateway, a.api_key, mode, r, sem, a.max_tokens) for r in rows))
            results.extend(res)
            print(f"[{mode}] {sum(r['ok'] for r in res)}/{len(res)} ok in {time.perf_counter() - t0:.1f}s")
    (out_dir / f"results-{stamp}.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")
    summary = summarize(results)
    (out_dir / f"summary-{stamp}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir}/results-{stamp}.jsonl  (run bench/judge.py on it for quality scores)")


if __name__ == "__main__":
    asyncio.run(main())
