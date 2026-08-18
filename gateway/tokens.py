"""Token counting. Uses tiktoken's cl100k_base as a stable approximation across models."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _enc():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - offline fallback
        return None


def count_tokens(text: str) -> int:
    enc = _enc()
    if enc is None:
        return max(1, len(text) // 4)
    return len(enc.encode(text))
