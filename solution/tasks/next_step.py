"""Task 1 helpers: top-k next-step prediction."""

from __future__ import annotations


def predict_next_step(adapter, partial_sequence: list[str], family: str | None = None, k: int = 5) -> list[str]:
    return adapter.top_k_next(partial_sequence, k=k, family=family)
