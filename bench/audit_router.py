"""Cross-tabulates the router's chosen tier against the prompt's human difficulty label.

python bench/audit_router.py bench/results/results-<stamp>.jsonl
"""

import collections
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
auto = [r for r in rows if r["mode"] == "auto" and r["ok"]]
ct = collections.Counter((r["difficulty"], r["requested_tier"]) for r in auto)
print("difficulty ->  small  medium  remote")
for d in ("easy", "medium", "hard"):
    print(f"{d:10s} -> " + "".join(f"{ct[(d, t)]:7d}" for t in ("small", "medium", "remote")))
scores = collections.defaultdict(list)
for r in auto:
    scores[r["category"]].append(r["route_score"])
print("\nmean route score by category:")
for c, v in sorted(scores.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
    print(f"  {c:13s} {sum(v) / len(v):.3f}")
