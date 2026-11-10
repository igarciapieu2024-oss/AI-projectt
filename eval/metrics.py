
"""Evaluation metrics for retrieval, refusals, and latency.

Owner: Elena.
"""
from __future__ import annotations

import math
from statistics import mean


def _as_groups(
    retrieved_ids: list[str] | list[list[str]],
) -> list[set[str]]:
    """Treat each retrieved source as one ranking position."""
    return [
        {item} if isinstance(item, str) else set(item)
        for item in retrieved_ids
    ]


def recall_at_k(
    retrieved_ids: list[str] | list[list[str]],
    gold_ids: list[str],
    k: int = 5,
) -> float:
    """Fraction of gold IDs covered by the first k sources."""
    gold = set(gold_ids)

    if not gold:
        return 0.0

    found = set()

    for group in _as_groups(retrieved_ids)[:k]:
        found.update(group & gold)

    return len(found) / len(gold)


def reciprocal_rank(
    retrieved_ids: list[str] | list[list[str]],
    gold_ids: list[str],
) -> float:
    """Reciprocal rank of the first relevant source."""
    gold = set(gold_ids)

    for rank, group in enumerate(
        _as_groups(retrieved_ids), start=1
    ):
        if group & gold:
            return 1.0 / rank

    return 0.0



def ndcg_at_k(
    retrieved_ids: list[str] | list[list[str]],
    gold_ids: list[str],
    k: int = 10,
) -> float:
    """Binary nDCG@k based on relevant retrieved sources."""
    if k <= 0:
        return 0.0

    gold = set(gold_ids)

    if not gold:
        return 0.0

    groups = _as_groups(retrieved_ids)

    # Each source counts as relevant (1) or irrelevant (0).
    relevance = [
        int(bool(group & gold))
        for group in groups
    ]

    dcg = sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(relevance[:k], start=1)
    )

    # Number of relevant sources, capped at k.
    ideal_hits = min(sum(relevance), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg else 0.0




def refusal_metrics(
    expected: list[bool],
    predicted: list[bool],
) -> dict[str, float]:
    """Calculate refusal precision, recall, and accuracy."""
    if len(expected) != len(predicted):
        raise ValueError("Expected and predicted lengths must match.")

    if not expected:
        return {
            "refusal_precision": 0.0,
            "refusal_recall": 0.0,
            "refusal_accuracy": 0.0,
        }

    tp = sum(e and p for e, p in zip(expected, predicted))
    fp = sum(not e and p for e, p in zip(expected, predicted))
    fn = sum(e and not p for e, p in zip(expected, predicted))

    correct = sum(
        e == p for e, p in zip(expected, predicted)
    )

    return {
        "refusal_precision": tp / (tp + fp) if tp + fp else 0.0,
        "refusal_recall": tp / (tp + fn) if tp + fn else 0.0,
        "refusal_accuracy": correct / len(expected),
    }


def average_latency(
    timings_ms: list[float],
) -> float:
    """Calculate average response latency in milliseconds."""
    return mean(timings_ms) if timings_ms else 0.0
