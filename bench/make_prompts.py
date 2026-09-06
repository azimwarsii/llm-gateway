"""Generates bench/prompts.jsonl: 200 prompts across 8 categories with a difficulty label.

Difficulty is the human prior used to sanity-check the router, not ground truth.
Regenerate with `python bench/make_prompts.py`; edit the templates to suit your domain.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(7)

CATEGORIES: dict[str, tuple[str, list[str]]] = {
    "factual": (
        "easy",
        [
            "What is the capital of {country}?",
            "Who wrote {book}?",
            "What year did {event} happen?",
            "Define the word '{word}' in one sentence.",
            "Convert {n} kilometers to miles.",
        ],
    ),
    "rewrite": (
        "easy",
        [
            "Rewrite this sentence in a formal tone: '{sentence}'",
            "Fix the grammar: '{bad_sentence}'",
            "Translate to Spanish: '{sentence}'",
            "Summarize in 10 words: '{paragraph}'",
        ],
    ),
    "extraction": (
        "easy",
        [
            "Extract the email addresses from this text and return them as a JSON list: '{text_with_emails}'",
            "From this order note, return JSON with fields item, quantity, city: '{order}'",
        ],
    ),
    "code_small": (
        "medium",
        [
            "Write a Python function that {code_task}.",
            "What does this Python code print?\n```python\n{snippet}\n```",
            "Write a SQL query to {sql_task}.",
            "Fix the bug in this function:\n```python\n{buggy}\n```",
        ],
    ),
    "reasoning": (
        "medium",
        [
            "{riddle}",
            "A train leaves at {t1} going {s1} km/h and another at {t2} going {s2} km/h in the same direction from the same station. When does the second catch up? Show your steps.",
            "Compare the tradeoffs of {opt_a} versus {opt_b} for {context}. Give a recommendation.",
        ],
    ),
    "code_large": (
        "hard",
        [
            "Design and implement a thread-safe LRU cache in Python with TTL support. Include tests and explain the concurrency model.",
            "Refactor this module for readability and performance, explaining each change:\n```python\n{long_snippet}\n```",
            "Write a Python async rate limiter using the token bucket algorithm, with a decorator API and unit tests.",
        ],
    ),
    "analysis": (
        "hard",
        [
            "Analyze the root cause of this incident and propose three remediations with tradeoffs:\n{incident}",
            "Critique this system design step by step and identify the two biggest scalability risks:\n{design}",
            "Plan a migration from {from_db} to {to_db} for a service handling {qps} QPS with zero downtime. Cover data sync, cutover, rollback.",
        ],
    ),
    "long_context": (
        "hard",
        [
            "Here is a long document. Answer the question at the end.\n\n{long_doc}\n\nQuestion: {long_q}",
        ],
    ),
}

FILL = {
    "country": ["France", "Japan", "Kenya", "Brazil", "Canada", "Egypt"],
    "book": ["Pride and Prejudice", "1984", "The Odyssey", "Things Fall Apart", "Dune"],
    "event": ["the first moon landing", "the fall of the Berlin Wall", "the founding of the UN"],
    "word": ["ephemeral", "ubiquitous", "idempotent", "latency", "quorum"],
    "n": ["5", "42", "100", "3.5"],
    "sentence": ["hey can u send me the report by tmrw", "the meeting got moved, fyi", "we're gonna ship it friday"],
    "bad_sentence": [
        "Their going to there house tomorow.",
        "Me and him has went to the store.",
        "Its to late too fix it.",
    ],
    "paragraph": [
        "The gateway routes each request to the cheapest model likely to answer well, falling back to stronger models when the cheap one fails or is unavailable, and records cost and latency for every call.",
        "Quantization reduces model weights from 16-bit to 4-bit integers, cutting memory roughly fourfold at a small accuracy cost, which lets larger models fit on consumer GPUs.",
    ],
    "text_with_emails": [
        "Contact ana@example.com or the team at support@corp.io; cc bob.smith@mail.org.",
        "Reach me at dev@site.dev, not at the old one (old@site.dev).",
    ],
    "order": ["Please send 3 blue mugs to Pune by Friday.", "Need 12 boxes of A4 paper delivered to Kolkata office."],
    "code_task": [
        "returns the n-th Fibonacci number iteratively",
        "checks whether a string is a palindrome ignoring punctuation",
        "merges two sorted lists",
        "counts word frequencies in a text file",
        "flattens a nested list",
    ],
    "snippet": [
        "x = [1, 2, 3]\nprint(x[::-1] + x[1:])",
        "d = {'a': 1}\nd.setdefault('b', []).append(2)\nprint(d)",
        "print(sum(i * i for i in range(4)))",
    ],
    "sql_task": [
        "find the top 5 customers by total order value",
        "list products that have never been ordered",
        "compute monthly revenue for 2024",
    ],
    "buggy": [
        "def avg(xs):\n    return sum(xs) / len(xs) if xs else None\n\nprint(avg([]) + 1)",
        "def find(items, target):\n    for i in range(len(items)):\n        if items[i] == target:\n            return i\n    return 0",
    ],
    "riddle": [
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Explain.",
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Show reasoning.",
        "I have 3 boxes: one with only apples, one with only oranges, one mixed, all labelled wrong. You may pick one fruit from one box. How do you relabel all correctly?",
    ],
    "t1": ["9:00", "10:00"],
    "s1": ["60", "80"],
    "t2": ["10:00", "11:30"],
    "s2": ["90", "120"],
    "opt_a": ["PostgreSQL", "Kafka", "REST", "monolith"],
    "opt_b": ["MongoDB", "RabbitMQ", "gRPC", "microservices"],
    "context": ["a small startup analytics product", "a high-throughput event pipeline", "an internal admin tool"],
    "long_snippet": [
        "import os,sys\ndef proc(f):\n    d=open(f).read()\n    r=[]\n    for l in d.split('\\n'):\n        if l!='' and l[0]!='#':\n            r.append(l.strip().split(','))\n    o={}\n    for x in r:\n        if x[0] in o: o[x[0]]=o[x[0]]+int(x[1])\n        else: o[x[0]]=int(x[1])\n    for k in o: print(k,o[k])\nproc(sys.argv[1])"
    ],
    "incident": [
        "At 14:02 p95 latency rose from 200ms to 9s. Redis CPU hit 100%. A deploy at 13:55 added a cache key per user session with no TTL. Memory hit maxmemory at 14:00 and eviction began. Traffic was normal.",
        "Nightly job failed 3 nights running with OOM. The job loads all rows of a 40M-row table into a pandas DataFrame before filtering. Table grew 20% last month.",
    ],
    "design": [
        "Single Postgres primary, no replicas. API servers call it directly. Sessions stored in Postgres. Images uploaded to the API server disk. Cron job on one API server sends emails. Expected growth: 10x users in a year.",
        "Every microservice has its own database; reporting queries join across services via nightly CSV exports over SFTP. Auth is a shared library copied into each repo.",
    ],
    "from_db": ["MySQL", "MongoDB"],
    "to_db": ["PostgreSQL", "CockroachDB"],
    "qps": ["2,000", "15,000"],
    "long_doc": [
        "\n".join(
            f"Section {i}: The {['ingestion', 'routing', 'caching', 'billing', 'auth', 'metrics'][i % 6]} service handles {['documents', 'requests', 'responses', 'invoices', 'tokens', 'samples'][i % 6]} and was last changed on 2026-0{(i % 9) + 1}-1{i % 3}. Its owner is {['Ana', 'Bo', 'Chen', 'Dee', 'Eli', 'Fay'][i % 6]} and its SLA is {[99.9, 99.5, 99.0][i % 3]}%."
            for i in range(60)
        )
    ],
    "long_q": [
        "Which sections have an SLA of 99.0% and who owns them?",
        "Which service was changed most recently, and what does it handle?",
    ],
}


def fill(t: str) -> str:
    out = t
    for k, vals in FILL.items():
        out = out.replace("{" + k + "}", random.choice(vals))
    return out


def main() -> None:
    per_cat = {
        "factual": 40,
        "rewrite": 30,
        "extraction": 20,
        "code_small": 35,
        "reasoning": 30,
        "code_large": 15,
        "analysis": 20,
        "long_context": 10,
    }
    rows = []
    for cat, n in per_cat.items():
        diff, templates = CATEGORIES[cat]
        for i in range(n):
            rows.append(
                {"id": f"{cat}-{i:03d}", "category": cat, "difficulty": diff, "prompt": fill(random.choice(templates))}
            )
    Path(__file__).with_name("prompts.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} prompts")


if __name__ == "__main__":
    main()
