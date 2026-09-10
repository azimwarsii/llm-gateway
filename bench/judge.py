"""Rubric-based quality judge.

Scores each answer 1-5 with a judge model (by default the gateway's `remote` tier), then
reports mean quality per mode and "quality parity" of auto/small vs. remote. Judging with a
model from the same family as the remote tier introduces some bias in its favour; the README
notes this. Use a different provider as judge if you can.

Usage:
  python bench/judge.py bench/results/results-<stamp>.jsonl
  python bench/judge.py results.jsonl --judge-url https://api.groq.com/openai/v1 --judge-model llama-3.3-70b-versatile --judge-key $GROQ_API_KEY
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
from pathlib import Path

import httpx

RUBRIC = """You are grading an AI assistant's answer. Score it from 1 to 5:
5 = fully correct, complete, well-organised; 4 = correct with minor omissions;
3 = partially correct or incomplete; 2 = mostly wrong or unhelpful; 1 = wrong, empty, or off-topic.
Respond with ONLY the integer score.

[PROMPT]
{prompt}

[ANSWER]
{answer}
"""


async def grade(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    key: str | None,
    headers_extra: dict,
    row: dict,
    sem: asyncio.Semaphore,
) -> int | None:
    if not row.get("ok"):
        return None
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 4,
        "messages": [{"role": "user", "content": RUBRIC.format(prompt=row["prompt"], answer=row.get("answer", ""))}],
    }
    headers = {**headers_extra}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    async with sem:
        for _ in range(3):
            try:
                r = await client.post(f"{url}/chat/completions", json=body, headers=headers, timeout=60)
                if r.status_code == 429:
                    await asyncio.sleep(3)
                    continue
                r.raise_for_status()
                txt = r.json()["choices"][0]["message"]["content"]
                m = re.search(r"[1-5]", txt)
                return int(m.group()) if m else None
            except httpx.HTTPError:
                await asyncio.sleep(1)
    return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--judge-url", default="http://localhost:8080/v1")
    ap.add_argument("--judge-model", default="remote")
    ap.add_argument("--judge-key", default=os.environ.get("JUDGE_KEY", "bench-key"))
    ap.add_argument("--concurrency", type=int, default=2)
    a = ap.parse_args()

    rows = [json.loads(line) for line in Path(a.results).read_text().splitlines() if line.strip()]
    sem = asyncio.Semaphore(a.concurrency)
    extra = {"X-Route-Tier": "remote"} if a.judge_url.startswith("http://localhost") else {}
    async with httpx.AsyncClient() as client:
        scores = await asyncio.gather(
            *(grade(client, a.judge_url, a.judge_model, a.judge_key, extra, r, sem) for r in rows)
        )
    for r, s in zip(rows, scores):
        r["quality"] = s
    out = Path(a.results).with_name(Path(a.results).stem + "-judged.jsonl")
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    by_mode: dict[str, list[float]] = {}
    by_mode_cat: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        if r.get("quality") is not None:
            by_mode.setdefault(r["mode"], []).append(r["quality"])
            by_mode_cat.setdefault((r["mode"], r["category"]), []).append(r["quality"])
    print("\nMean quality (1-5) by mode:")
    for m, v in by_mode.items():
        print(f"  {m:8s} {statistics.mean(v):.2f}  (n={len(v)})")
    if "remote" in by_mode:
        base = statistics.mean(by_mode["remote"])
        for m, v in by_mode.items():
            if m != "remote":
                print(f"  parity {m} vs remote: {statistics.mean(v) / base * 100:.1f}%")
    print("\nBy category:")
    cats = sorted({c for _, c in by_mode_cat})
    modes = list(by_mode)
    print("  " + "category".ljust(14) + "".join(m.rjust(9) for m in modes))
    for c in cats:
        print(
            "  "
            + c.ljust(14)
            + "".join(
                (f"{statistics.mean(by_mode_cat[(m, c)]):.2f}" if (m, c) in by_mode_cat else "-").rjust(9)
                for m in modes
            )
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
