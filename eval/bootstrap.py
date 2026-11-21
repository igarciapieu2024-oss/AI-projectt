# Confidence intervals for the evaluation metrics, by bootstrap. With 30 test questions a score such as 73%
# could easily be a few points higher or lower on other questions, so the interval shows how uncertain it is.
# Bootstrap resampling estimates that uncertainty without assuming any distribution for the scores. To compare
# two systems, the differences are resampled question by question (paired), so both systems are always
# compared on the same questions.
#   python -m eval.bootstrap --results-a <metrics of system A> [--results-b <metrics of system B>] --metric recall_at_5
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean


# The p-th percentile of the values (p from 0 to 100), interpolating between the two closest values.
def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of empty values.")

    if not 0 <= p <= 100:
        raise ValueError("Percentile must be between 0 and 100.")

    # The position of the percentile in the sorted list; when it falls between two values, they are mixed
    # in proportion to the distance.
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100

    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return (
        ordered[lower] * (1 - fraction)
        + ordered[upper] * fraction
    )


# The mean of the values and its confidence interval (95% by default). The values are resampled with
# replacement n_bootstrap times (2000), the mean of each sample is kept, and the interval is the range
# between the 2.5th and 97.5th percentiles of those means. The fixed seed gives the same interval every time.
# The interval only shows the uncertainty of having few questions, not errors of the grading itself.
def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    if not values:
        raise ValueError("Values cannot be empty.")

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")

    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")

    rng = random.Random(seed)
    n = len(values)
    estimates = []

    # Each sample has as many values as the original, drawn at random with replacement.
    for _ in range(n_bootstrap):
        sample = [
            values[rng.randrange(n)]
            for _ in range(n)
        ]
        estimates.append(mean(sample))

    # The share left out on each side of the interval (2.5% for a 95% interval).
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


# Compare two systems on the same questions: the difference B minus A of one metric, question by question,
# with its confidence interval. If the interval does not include 0, B is better (or worse) beyond what the
# small number of questions can explain. Pairing by question_id matters: some questions are hard for both
# systems, and resampling the two systems separately would ignore that.
def paired_bootstrap(
    results_a: list[dict],
    results_b: list[dict],
    metric: str,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    a = {row["question_id"]: row for row in results_a}
    b = {row["question_id"]: row for row in results_b}

    # Each question must appear once in each system, and both systems must have the same questions.
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

        # A metric that does not apply to the question in either system (for example recall on an
        # unanswerable question) is skipped.
        if value_a is None and value_b is None:
            continue

        # Defined in one system but not in the other: the results do not match.
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


# Read a JSON file with one record of metrics per question.
def load_results(path: str | Path) -> list[dict]:
    data = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    if not isinstance(data, list):
        raise ValueError(
            "Expected a JSON list of evaluation results."
        )

    return data


# Command line: with one results file, the interval of its metric; with two, the paired difference.
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


# Runs only when the file is executed directly.
if __name__ == "__main__":
    main()
