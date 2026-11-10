
"""Bootstrap confidence intervals for evaluation metrics.

Supports single-system confidence intervals and paired comparisons.

Owner: Elena.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean


def percentile(values: list[float], p: float) -> float:
    """Compute a percentile using linear interpolation."""
    if not values:
        raise ValueError("Cannot compute percentile of empty values.")

    if not 0 <= p <= 100:
        raise ValueError("Percentile must be between 0 and 100.")

    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100

    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return (
        ordered[lower] * (1 - fraction)
        + ordered[upper] * fraction
    )


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Estimate a confidence interval for a sample mean."""
    if not values:
        raise ValueError("Values cannot be empty.")

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")

    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")

    rng = random.Random(seed)
    n = len(values)
    estimates = []

    for _ in range(n_bootstrap):
        sample = [
            values[rng.randrange(n)]
            for _ in range(n)
        ]
        estimates.append(mean(sample))

    alpha = (1 - confidence) / 2

    return {
        "mean": mean(values),
        "ci_lower": percentile(estimates, alpha * 100),
        "ci_upper": percentile(
            estimates, (1 - alpha) * 100
        ),
        "confidence": confidence,
        "n_samples": n,
        "n_bootstrap": n_bootstrap,
    }


def paired_bootstrap(
    results_a: list[dict],
    results_b: list[dict],
    metric: str,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compare configurations using matched question IDs.

    Difference = system B minus system A.
    """
    a = {row["question_id"]: row for row in results_a}
    b = {row["question_id"]: row for row in results_b}

    if len(a) != len(results_a) or len(b) != len(results_b):
        raise ValueError("Duplicate question IDs are not allowed.")

    if set(a) != set(b):
        raise ValueError(
            "Paired bootstrap requires identical question IDs."
        )

    ids = sorted(a)

    differences = []

    for question_id in ids:
        value_a = a[question_id].get(metric)
        value_b = b[question_id].get(metric)

        # Skip undefined metrics in both configurations.
        if value_a is None and value_b is None:
            continue

        if value_a is None or value_b is None:
            raise ValueError(
                f"Metric missing for question {question_id}."
            )

        differences.append(float(value_b) - float(value_a))

    if not differences:
        raise ValueError(
            "No valid paired observations for this metric."
        )

    result = bootstrap_ci(
        differences,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )

    return {
        "metric": metric,
        "mean_difference_b_minus_a": result["mean"],
        "ci_lower": result["ci_lower"],
        "ci_upper": result["ci_upper"],
        "confidence": confidence,
        "n_pairs": result["n_samples"],
        "n_bootstrap": n_bootstrap,
    }


def load_results(path: str | Path) -> list[dict]:
    """Load individual evaluation results from JSON."""
    data = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    if not isinstance(data, list):
        raise ValueError(
            "Expected a JSON list of evaluation results."
        )

    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap evaluation metrics."
    )

    parser.add_argument("--results-a", required=True)
    parser.add_argument("--results-b")
    parser.add_argument("--metric", default="recall_at_5")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    results_a = load_results(args.results_a)

    if args.results_b:
        results_b = load_results(args.results_b)

        result = paired_bootstrap(
            results_a,
            results_b,
            metric=args.metric,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
    else:
        values = [
            float(row[args.metric])
            for row in results_a
            if row.get(args.metric) is not None
        ]

        result = bootstrap_ci(
            values,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
