"""Complexity-based router.

Scores a request in [0, 1] from cheap, explainable features, then maps the score to a
tier. Explainability matters here: every routing decision is returned in the response
metadata so the benchmark can audit it.

Upgrade path: replace `score()` with a small fine-tuned classifier trained on the
benchmark's per-tier win/loss labels. The interface stays the same.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import RouterConfig, Tier
from .schemas import ChatCompletionRequest
from .tokens import count_tokens

REASONING_HINTS = re.compile(
    r"\b(prove|derive|step[- ]by[- ]step|analy[sz]e|compare|trade[- ]?offs?|design|architect|"
    r"optimi[sz]e|debug|refactor|explain why|root cause|evaluate|critique|plan)\b",
    re.IGNORECASE,
)
CODE_HINTS = re.compile(
    r"```|\bdef |\bclass |\bfunction\b|\bSELECT\b.+\bFROM\b|\{.*\}|;\s*$", re.IGNORECASE | re.DOTALL
)
MATH_HINTS = re.compile(r"[=<>^]|\\frac|\bintegral\b|\bderivative\b|\bmatrix\b|\d+\s*[\+\-\*/]\s*\d+")
SIMPLE_HINTS = re.compile(
    r"^(what is|who is|define|translate|summari[sz]e|rewrite|fix typo|capital of|convert|list)\b", re.IGNORECASE
)


@dataclass
class RouteDecision:
    tier: Tier
    score: float
    features: dict[str, float] = field(default_factory=dict)
    reason: str = ""


def score(req: ChatCompletionRequest) -> tuple[float, dict[str, float]]:
    text = req.prompt_text()
    last_user = next((m.text() for m in reversed(req.messages) if m.role == "user"), text)
    n_tokens = count_tokens(text)
    n_turns = sum(1 for m in req.messages if m.role != "system")
    has_tools = bool(getattr(req, "tools", None) or getattr(req, "functions", None))
    wants_json = bool(getattr(req, "response_format", None))

    f = {
        "len": min(n_tokens / 1500, 1.0),  # long prompts need more capacity
        "turns": min(n_turns / 10, 1.0),
        "reasoning": 1.0 if REASONING_HINTS.search(last_user) else 0.0,
        "code": 1.0 if CODE_HINTS.search(text) else 0.0,
        "math": 1.0 if MATH_HINTS.search(last_user) else 0.0,
        "simple": 1.0 if SIMPLE_HINTS.search(last_user.strip()) else 0.0,
        "tools": 1.0 if has_tools else 0.0,
        "json": 1.0 if wants_json else 0.0,
        "max_tokens": min((req.max_tokens or 256) / 2048, 1.0),
    }
    s = (
        0.25 * f["len"]
        + 0.10 * f["turns"]
        + 0.25 * f["reasoning"]
        + 0.15 * f["code"]
        + 0.10 * f["math"]
        + 0.15 * f["tools"]
        + 0.05 * f["json"]
        + 0.10 * f["max_tokens"]
        - 0.25 * f["simple"]
    )
    return max(0.0, min(1.0, s)), f


def route(req: ChatCompletionRequest, cfg: RouterConfig, override: str | None = None) -> RouteDecision:
    if override and cfg.allow_tier_override and override in ("small", "medium", "remote"):
        return RouteDecision(tier=override, score=-1.0, reason="header override")  # type: ignore[arg-type]
    # Explicit model name in the request pins a tier ("small"/"medium"/"remote") or a backend.
    if req.model in ("small", "medium", "remote"):
        return RouteDecision(tier=req.model, score=-1.0, reason="model field")  # type: ignore[arg-type]
    s, f = score(req)
    if s < cfg.small_max:
        tier: Tier = "small"
    elif s < cfg.medium_max:
        tier = "medium"
    else:
        tier = "remote"
    return RouteDecision(tier=tier, score=round(s, 3), features=f, reason="complexity score")
