"""Task 2 helpers: sequence completion."""

from __future__ import annotations


def complete_sequence(adapter, partial_sequence: list[str], family: str | None = None) -> list[str]:
    return adapter.complete(partial_sequence, family=family)
