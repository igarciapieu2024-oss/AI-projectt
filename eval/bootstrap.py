# Confidence intervals for the evaluation metrics, by bootstrap. With 30 test questions, a score such as 73% could
# easily be a few points higher or lower on other questions, so the interval shows how uncertain it is. Bootstrap
# resampling estimates that uncertainty without assuming any distribution for the scores. To compare two systems, we
# resample the differences question by question (paired), so both systems are always compared on the same questions.
#   python -m eval.bootstrap --results-a <metrics of system A> [--results-b <metrics of system B>] --metric
# recall_at_5
from __future__ import annotations

# argparse reads the options, json reads and prints the results, random draws the samples and mean averages them.
import argparse
import json
import random
from pathlib import Path
from statistics import mean


# The p-th percentile of the values (p from 0 to 100), interpolating between the two closest values.
def percentile(values: list[float], p: float) -> float:
    # We need at least one value, and a percentile between 0 and 100.
    if not values:
        raise ValueError("Cannot compute percentile of empty values.")

    if not 0 <= p <= 100:
        raise ValueError("Percentile must be between 0 and 100.")

    # The position of the percentile in the sorted list; when it falls between two values, we mix them in proportion
    # to the distance.
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100

    # The two values around that position, and how far the position is between them.
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    # A weighted mix of the two values.
    return (
        ordered[lower] * (1 - fraction)
        + ordered[upper] * fraction
    )


# The mean of the values and its confidence interval (95% by default). We resample the values with replacement
# n_bootstrap times (2000), keep the mean of each sample, and the interval is the range between the 2.5th and the
# 97.5th percentiles of those means. The fixed seed gives the same interval every time. The interval only shows the
# uncertainty of having few questions, not errors in the grading itself.
def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    # The three checks: some values, at least one resample and a confidence between 0 and 1.
    if not values:
        raise ValueError("Values cannot be empty.")

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")

    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")

    # The random generator with a fixed seed, the number of values and the mean of each sample.
    rng = random.Random(seed)
    n = len(values)
    estimates = []

    # Each sample has as many values as the original, drawn at random with replacement (the same value can be drawn
    # more than once).
    for _ in range(n_bootstrap):
        sample = [
            values[rng.randrange(n)]
            for _ in range(n)
        ]
        estimates.append(mean(sample))

    # The share left out on each side of the interval (2.5% for a 95% interval).
    alpha = (1 - confidence) / 2

    # The mean of the real values, and the two percentiles of the sample means as the interval.
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


# Compare two systems on the same questions: the difference B minus A of one metric, question by question, with its
# confidence interval. If the interval doesn't include 0, B is better (or worse) beyond what the small number of
# questions can explain. Pairing by question_id matters: some questions are hard for both systems, and resampling the
# two systems separately would ignore that.
def paired_bootstrap(
    results_a: list[dict],
    results_b: list[dict],
    metric: str,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    # The rows of each system by question id.
    a = {row["question_id"]: row for row in results_a}
    b = {row["question_id"]: row for row in results_b}

    # Each question has to appear once in each system, and both systems need the same questions.
    if len(a) != len(results_a) or len(b) != len(results_b):
        raise ValueError("Duplicate question IDs are not allowed.")

    if set(a) != set(b):
        raise ValueError(
            "Paired bootstrap requires identical question IDs."
        )

    # The questions in a fixed order, and the difference of each one.
    ids = sorted(a)

    differences = []

    for question_id in ids:
        # The value of the metric in each system.
        value_a = a[question_id].get(metric)
        value_b = b[question_id].get(metric)

        # A metric that doesn't apply to the question in either system (for example recall on an unanswerable
        # question) is skipped.
        if value_a is None and value_b is None:
            continue

        # Defined in one system but not in the other: the results don't match.
        if value_a is None or value_b is None:
            raise ValueError(
                f"Metric missing for question {question_id}."
            )

        # The difference of this question: B minus A.
        differences.append(float(value_b) - float(value_a))

    # Without any pair there is nothing to compare.
    if not differences:
        raise ValueError(
            "No valid paired observations for this metric."
        )

    # The interval of the differences, calculated like any other list of values.
    result = bootstrap_ci(
        differences,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )

    # The same result, with names that say it is a difference.
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
    # Read the file as JSON.
    data = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    # It has to be a list, one record per question.
    if not isinstance(data, list):
        raise ValueError(
            "Expected a JSON list of evaluation results."
        )

    return data


# Command line: with one results file, the interval of its metric; with two, the paired difference.
def main() -> None:
    # The options: one or two results files, the metric, the number of resamples and the seed.
    parser = argparse.ArgumentParser(
        description="Bootstrap evaluation metrics."
    )

    parser.add_argument("--results-a", required=True)
    parser.add_argument("--results-b")
    parser.add_argument("--metric", default="recall_at_5")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # The results of system A.
    results_a = load_results(args.results_a)

    # With a second file: the paired difference between the two systems.
    if args.results_b:
        results_b = load_results(args.results_b)

        result = paired_bootstrap(
            results_a,
            results_b,
            metric=args.metric,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
    # With one file: the interval of the metric, over the questions that have it.
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

    # Print the result as JSON.
    print(json.dumps(result, indent=2))


# Runs only when the file is executed directly.
if __name__ == "__main__":
    main()
