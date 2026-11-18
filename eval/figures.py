"""Generate held-out test figures for B0, B1, and Final v1.2.

All values come from saved summary files; this module never reruns the
generator or judge. B0 is omitted from the retrieval chart because a
no-RAG model has no retrieval step (N/A, rather than zero recall).

Usage:
    python -m eval.figures

Owner: Elena.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


B0_RUN = Path("eval/runs/20261116T193556Z_b0_no_rag_test")
B1_RUN = Path("eval/runs/20261116T193243Z_b1_v1_test")
FINAL_RUN = Path("eval/runs/20261116T192633Z_final_test")

OUTPUT = Path("reports/figures/test")


def load_json(path: Path) -> dict:
    """Load the stored metric or judge summary for a fixed evaluation run."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def grouped_chart(
    title: str,
    labels: list[str],
    b1_values: list[float],
    final_values: list[float],
    filename: str,
    b0_values: list[float] | None = None,
    ylabel: str = "Score",
    percentage: bool = True,
    higher_is_better: bool = True,
) -> None:
    """Draw comparable bars for two or three systems and label their values.

    B0 is optional so retrieval metrics can compare only RAG systems.
    Fractions are formatted as percentages; latency uses seconds instead.
    """

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))

    if b0_values is None:
        # No-RAG has no meaningful Recall@K or MRR, so do not plot a zero.
        width = 0.33
        positions = [
            (x - width / 2, b1_values, "B1 Baseline", "#A9B8CC"),
            (x + width / 2, final_values, "Final v1.2", "#3977AD"),
        ]
    else:
        width = 0.25
        positions = [
            (x - width, b0_values, "B0 No RAG", "#E5A45C"),
            (x, b1_values, "B1 Baseline", "#A9B8CC"),
            (x + width, final_values, "Final v1.2", "#3977AD"),
        ]

    bar_groups = []

    for position, values, name, color in positions:
        bars = ax.bar(
            position,
            values,
            width,
            label=name,
            color=color,
        )
        bar_groups.append(bars)

    all_values = (b0_values or []) + b1_values + final_values
    maximum = max(all_values)

    if percentage:
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.0%}")
        )
    else:
        ax.set_ylim(0, maximum * 1.2 if maximum else 1)

    for bars in bar_groups:
        for bar in bars:
            value = bar.get_height()

            label = (
                f"{value:.1%}"
                if percentage
                else f"{value:.2f}"
            )

            ax.annotate(
                label,
                (
                    bar.get_x() + bar.get_width() / 2,
                    value,
                ),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)

    fig.tight_layout()

    OUTPUT.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        OUTPUT / filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", OUTPUT / filename)


def main() -> None:
    """Build retrieval, correctness, refusal, and latency figures from test."""
    b0_metrics = load_json(B0_RUN / "metrics_summary.json")
    b0_judge = load_json(B0_RUN / "judge_summary.json")
    b1_metrics = load_json(B1_RUN / "metrics_summary.json")
    final_metrics = load_json(FINAL_RUN / "metrics_summary.json")

    b1_judge = load_json(B1_RUN / "judge_summary.json")
    final_judge = load_json(FINAL_RUN / "judge_summary.json")

    # Figure 1: Retrieval performance

    grouped_chart(
        title="Retrieval Performance",
        labels=["Recall@K", "MRR@K"],
        b1_values=[
            b1_metrics["recall_at_k"],
            b1_metrics["mrr_at_k"],
        ],
        final_values=[
            final_metrics["recall_at_k"],
            final_metrics["mrr_at_k"],
        ],
        filename="retrieval_performance.png",
    )

    # Figure 2: Answer correctness

    grouped_chart(
        title="Answer Correctness",
        labels=["Exact Correctness", "Mean Correctness"],
        b1_values=[
            b1_judge["exact_correct_rate"],
            b1_judge["mean_correctness_score"],
        ],
        final_values=[
            final_judge["exact_correct_rate"],
            final_judge["mean_correctness_score"],
        ],
        filename="answer_correctness.png",
        b0_values=[
            b0_judge["exact_correct_rate"],
            b0_judge["mean_correctness_score"],
        ],
    )

    # Figure 3: Refusal performance (automatic metrics)

    grouped_chart(
        title="Refusal Performance (Automatic Evaluation)",
        labels=[
            "Refusal Accuracy",
            "Correct Unanswerable Refusal",
        ],
        b1_values=[
            b1_metrics["refusal_accuracy"],
            b1_metrics["correct_unanswerable_refusal_rate"],
        ],
        final_values=[
            final_metrics["refusal_accuracy"],
            final_metrics["correct_unanswerable_refusal_rate"],
        ],
        filename="refusal_performance.png",
                b0_values=[
            b0_metrics["refusal_accuracy"],
            b0_metrics["correct_unanswerable_refusal_rate"],
        ],
    )

    # Figure 4: Latency

    grouped_chart(
        title="Mean Response Latency",
        labels=["Mean Latency"],
        b1_values=[
            b1_metrics["mean_latency_ms"] / 1000,
        ],
        final_values=[
            final_metrics["mean_latency_ms"] / 1000,
        ],
        filename="latency_comparison.png",
        ylabel="Seconds",
        percentage=False,
                b0_values=[
            b0_metrics["mean_latency_ms"] / 1000,
        ],
    )

    print("\n========== FIGURES COMPLETE ==========")
    print("Output directory:", OUTPUT)
    print("Total figures: 4")
    print("======================================")


if __name__ == "__main__":
    main()
