"""Task 3 helpers: anomaly scoring and rule attribution."""

from __future__ import annotations

from solution.eval import rules


def anomaly_score(adapter, sequence: list[str], family: str | None = None) -> float:
    return -adapter.seq_log_prob(sequence, family=family)


def attribute_rule(sequence: list[str], family: str | None = None) -> str | None:
    return rules.attribute_anomaly(sequence, family=family)
