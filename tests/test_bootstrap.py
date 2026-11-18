
"""Unit tests for bootstrap evaluation.

Owner: Elena.
"""
import json

import pytest

from eval.bootstrap import (
    bootstrap_ci,
    paired_bootstrap,
    percentile,
    load_results,
)


def test_percentile():
    """Percentiles use linear interpolation."""
    assert percentile([0, 10, 20], 50) == 10
    assert percentile([0, 10, 20], 25) == 5


def test_bootstrap_mean():
    """Bootstrap preserves the observed sample mean."""
    result = bootstrap_ci(
        [0.0, 0.5, 1.0],
        n_bootstrap=100,
        seed=42,
    )

    assert result["mean"] == 0.5
    assert result["ci_lower"] <= result["mean"]
    assert result["ci_upper"] >= result["mean"]


def test_bootstrap_reproducibility():
    """A fixed random seed produces identical results."""
    first = bootstrap_ci([0, 1, 1, 0], seed=42)
    second = bootstrap_ci([0, 1, 1, 0], seed=42)

    assert first == second


def test_bootstrap_constant_values():
    """Identical observations produce a zero-width interval."""
    result = bootstrap_ci([0.8, 0.8, 0.8])

    assert result["mean"] == pytest.approx(0.8)
    assert result["ci_lower"] == pytest.approx(0.8)
    assert result["ci_upper"] == pytest.approx(0.8)


def test_paired_bootstrap():
    """Compare two systems using the same question IDs."""
    results_a = [
        {"question_id": "Q1", "recall_at_5": 0.0},
        {"question_id": "Q2", "recall_at_5": 0.5},
    ]

    results_b = [
        {"question_id": "Q1", "recall_at_5": 1.0},
        {"question_id": "Q2", "recall_at_5": 1.0},
    ]

    result = paired_bootstrap(
        results_a,
        results_b,
        metric="recall_at_5",
        n_bootstrap=100,
    )

    assert result["mean_difference_b_minus_a"] == 0.75
    assert result["n_pairs"] == 2
    assert result["ci_lower"] <= 0.75
    assert result["ci_upper"] >= 0.75


def test_paired_bootstrap_mismatched_ids():
    """Both systems must evaluate the same questions."""
    a = [{"question_id": "Q1", "recall_at_5": 0.5}]
    b = [{"question_id": "Q2", "recall_at_5": 0.5}]

    with pytest.raises(ValueError, match="identical question IDs"):
        paired_bootstrap(a, b, metric="recall_at_5")


def test_invalid_bootstrap_input():
    """Invalid bootstrap settings raise meaningful errors."""
    with pytest.raises(ValueError):
        bootstrap_ci([])

    with pytest.raises(ValueError):
        bootstrap_ci([1.0], n_bootstrap=0)

    with pytest.raises(ValueError):
        bootstrap_ci([1.0], confidence=1.5)


def test_load_results(tmp_path):
    """Read evaluation results from a JSON file."""
    path = tmp_path / "results.json"

    expected = [
        {"question_id": "Q1", "recall_at_5": 1.0}
    ]

    path.write_text(json.dumps(expected), encoding="utf-8")

    assert load_results(path) == expected
