
"""Unit tests for retrieval evaluation metrics.

Owner: Elena.
"""
import pytest

from eval.metrics import (
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    refusal_metrics,
    average_latency,
)


def test_recall_at_five():
    """Recall@5 finds the relevant sources."""
    retrieved = ["A", "B", "C", "D", "E"]
    gold = ["B", "C"]

    assert recall_at_k(retrieved, gold, k=5) == 1.0


def test_recall_partial():
    """Recall is partial when only some gold sources are found."""
    retrieved = ["A", "B", "D"]
    gold = ["B", "C"]

    assert recall_at_k(retrieved, gold, k=5) == 0.5


def test_reciprocal_rank():
    """MRR component uses the first relevant source position."""
    retrieved = ["A", "B", "C"]
    gold = ["B", "C"]

    assert reciprocal_rank(retrieved, gold) == 0.5


def test_ndcg():
    """nDCG rewards relevant sources appearing earlier."""
    retrieved = ["A", "B", "C"]
    gold = ["B", "C"]

    assert ndcg_at_k(retrieved, gold, k=10) == pytest.approx(
        0.6934, abs=0.001
    )


def test_no_relevant_sources():
    """Metrics return zero when no relevant source is retrieved."""
    retrieved = ["A", "B"]
    gold = ["C"]

    assert recall_at_k(retrieved, gold) == 0.0
    assert reciprocal_rank(retrieved, gold) == 0.0
    assert ndcg_at_k(retrieved, gold) == 0.0


def test_empty_gold_ids():
    """An empty gold set must not cause division by zero."""
    retrieved = ["A", "B"]
    gold = []

    assert recall_at_k(retrieved, gold) == 0.0
    assert reciprocal_rank(retrieved, gold) == 0.0
    assert ndcg_at_k(retrieved, gold) == 0.0

def test_multiple_ids_per_source():
    """A source covering multiple IDs counts as one retrieval position."""
    retrieved = [
        ["A", "B"],   # Source 1
        ["C"],        # Source 2
        ["D", "E"],   # Source 3
    ]
    gold = ["B", "E"]

    assert recall_at_k(retrieved, gold, k=1) == 0.5
    assert recall_at_k(retrieved, gold, k=3) == 1.0
    assert reciprocal_rank(retrieved, gold) == 1.0


def test_multiple_ids_source_position():
    """MRR uses source position, not individual ID position."""
    retrieved = [
        ["A", "B"],  # Source 1
        ["C", "D"],  # Source 2
    ]
    gold = ["D"]

    assert reciprocal_rank(retrieved, gold) == 0.5
    assert recall_at_k(retrieved, gold, k=1) == 0.0
    assert recall_at_k(retrieved, gold, k=2) == 1.0

def test_multiple_ids_per_source():
    """One retrieved source may cover multiple gold IDs."""
    retrieved = [
        ["A", "B"],
        ["C"],
        ["D", "E"],
    ]
    gold = ["B", "E"]

    assert recall_at_k(retrieved, gold, k=1) == 0.5
    assert recall_at_k(retrieved, gold, k=3) == 1.0
    assert reciprocal_rank(retrieved, gold) == 1.0


def test_multiple_ids_source_position():
    """Ranking is based on sources, not individual IDs."""
    retrieved = [
        ["A", "B"],
        ["C", "D"],
    ]
    gold = ["D"]

    assert reciprocal_rank(retrieved, gold) == 0.5
    assert recall_at_k(retrieved, gold, k=1) == 0.0
    assert recall_at_k(retrieved, gold, k=2) == 1.0

def test_refusal_metrics():
    """Calculate precision, recall and accuracy of refusals."""
    expected = [True, True, False, False]
    predicted = [True, False, True, False]

    result = refusal_metrics(expected, predicted)

    assert result["refusal_precision"] == 0.5
    assert result["refusal_recall"] == 0.5
    assert result["refusal_accuracy"] == 0.5


def test_refusal_metrics_perfect():
    """All correctly classified refusals score 1.0."""
    result = refusal_metrics(
        expected=[True, False, True],
        predicted=[True, False, True],
    )

    assert result["refusal_precision"] == 1.0
    assert result["refusal_recall"] == 1.0
    assert result["refusal_accuracy"] == 1.0


def test_refusal_metrics_invalid_lengths():
    """Different input lengths should raise an error."""
    with pytest.raises(ValueError):
        refusal_metrics(
            expected=[True, False],
            predicted=[True],
        )


def test_average_latency():
    """Calculate mean response time in milliseconds."""
    assert average_latency([100.0, 200.0, 300.0]) == 200.0


def test_average_latency_empty():
    """Empty latency lists should return zero."""
    assert average_latency([]) == 0.0
