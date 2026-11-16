"""Generate evaluation figures comparing B1 and Final.

Usage:
    python -m eval.figures

Owner: Elena.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


B1_RUN = Path("eval/runs/20261114T150015Z_b1_v1_dev")
FINAL_RUN = Path("eval/runs/20261114T190438Z_final_dev")

OUTPUT = Path("reports/figures")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def grouped_chart(
    title: str,
    labels: list[str],
    b1_values: list[float],
    final_values: list[float],
    filename: str,
    ylabel: str = "Score",
    percentage: bool = True,
    higher_is_better: bool = True,
) -> None:

    x = np.arange(len(labels))
    width = 0.33

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_b1 = ax.bar(
        x - width / 2,
        b1_values,
        width,
        label="B1 Baseline",
        color="#A9B8CC",
    )

    bars_final = ax.bar(
        x + width / 2,
        final_values,
        width,
        label="Final v1.1",
        color="#3977AD",
    )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=18)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)

    maximum = max(b1_values + final_values)

    if percentage:
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.0%}")
        )
    else:
        ax.set_ylim(0, maximum * 1.2 if maximum else 1)

    for bars in [bars_b1, bars_final]:
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
    )

    print("\n========== FIGURES COMPLETE ==========")
    print("Output directory:", OUTPUT)
    print("Total figures: 4")
    print("======================================")


if __name__ == "__main__":
    main()