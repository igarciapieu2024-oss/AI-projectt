# Draw four bar charts that compare three systems on the test questions: B0 (the LLM alone, no RAG), B1 (the
# baseline RAG) and "Final v1.2" (the version of the final system on 16 November, before the changes of
# 18 November). The values are read from the saved summary files of three fixed runs; nothing is run again.
#   python -m eval.figures
# The charts are saved in reports/figures/test/.

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# The three runs compared (folders written by eval/run_eval.py and eval/judge.py).
B0_RUN = Path("eval/runs/20261116T193556Z_b0_no_rag_test")
B1_RUN = Path("eval/runs/20261116T193243Z_b1_v1_test")
FINAL_RUN = Path("eval/runs/20261116T192633Z_final_test")

OUTPUT = Path("reports/figures/test")


# Read one summary file of a run (metrics_summary.json or judge_summary.json).
def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# Draw one chart: a group of bars per metric, one bar per system, with its value written on top, and save it.
# B0 is optional because the retrieval chart only compares the two RAG systems. Shares are shown as
# percentages; the latency chart shows seconds.
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

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))

    if b0_values is None:
        # Two bars per group. The LLM alone has no search step, so it has no recall or MRR; plotting 0 would
        # suggest it searched and failed.
        width = 0.33
        positions = [
            (x - width / 2, b1_values, "B1 Baseline", "#A9B8CC"),
            (x + width / 2, final_values, "Final v1.2", "#3977AD"),
        ]
    # Three bars per group.
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

    # The height of the axis leaves room for the value labels above the bars.
    all_values = (b0_values or []) + b1_values + final_values
    maximum = max(all_values)

    if percentage:
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.0%}")
        )
    else:
        ax.set_ylim(0, maximum * 1.2 if maximum else 1)

    # Write each value on top of its bar.
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


# Read the summaries of the three runs and draw the retrieval, correctness, refusal and latency charts.
def main() -> None:
    b0_metrics = load_json(B0_RUN / "metrics_summary.json")
    b0_judge = load_json(B0_RUN / "judge_summary.json")
    b1_metrics = load_json(B1_RUN / "metrics_summary.json")
    final_metrics = load_json(FINAL_RUN / "metrics_summary.json")

    b1_judge = load_json(B1_RUN / "judge_summary.json")
    final_judge = load_json(FINAL_RUN / "judge_summary.json")

    # Figure 1: retrieval (B1 and the final system only).

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

    # Figure 2: answer correctness from the judge (fully correct share and average score).

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

    # Figure 3: refusals, from the automatic metrics.

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

    # Figure 4: average time per answer, in seconds.

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


# Runs only when the file is executed directly.
if __name__ == "__main__":
    main()
