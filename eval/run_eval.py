
"""Run evaluation and save individual and aggregate results.

Usage:
    python -m eval.run_eval --config configs/b1_v1.yaml --split dev --fake

Owner: Elena.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from eval.metrics import (
    average_latency,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    refusal_metrics,
)
from src.config import PATHS, load_config
from src.schemas import Filters


def evaluate(
    config_path: str,
    split: str,
    use_fake: bool = False,
    questions_path: Path | None = None,
) -> Path:
    """Evaluate questions and save detailed and aggregate results."""

    # Protect the final test set.
    if split != "dev":
        raise ValueError(
            "The test set is locked. Only dev is allowed for now."
        )

    cfg = load_config(config_path)

    # Select the answer function.
    if use_fake:
        from src.generation.fake import answer
    else:
        from src.generation.answer import answer

    # Allow an isolated question dataset for unit tests.
    questions_path = (
        Path(questions_path)
        if questions_path is not None
        else PATHS["questions"]
    )

    with questions_path.open(
        newline="", encoding="utf-8"
    ) as file:
        questions = [
            row for row in csv.DictReader(file)
            if row["split"].strip() == split
        ]

    if not questions:
        raise ValueError(
            "No dev questions found. Assign dev splits before evaluation."
        )

    results = []

    for row in questions:
        history = (
            [{"role": "user", "content": row["turn1"]}]
            if row["turn1"].strip()
            else []
        )

        filters = Filters(
            course_code=row["course_code"].strip() or None
        )

        result = answer(
            question=row["question"],
            history=history,
            filters=filters,
            cfg=cfg,
        )

        gold_ids = sorted({
            value.strip()
            for value in row["gold_ids"].split(";")
            if value.strip()
        })

        # Each group represents one retrieved source.
        retrieved_groups = [
            hit.chunk.covers
            for hit in result.context_hits
        ]

        retrieved_ids = [
            source_id
            for group in retrieved_groups
            for source_id in group
        ]

        # Retrieval metrics are defined for questions with
        # gold evidence, not unanswerable questions.
        has_gold = bool(gold_ids)

        recall5 = (
            recall_at_k(retrieved_groups, gold_ids, k=5)
            if has_gold else None
        )

        rr = (
            reciprocal_rank(retrieved_groups, gold_ids)
            if has_gold else None
        )

        ndcg10 = (
            ndcg_at_k(retrieved_groups, gold_ids, k=10)
            if has_gold else None
        )

        expected_refusal = row["answerable"].strip() == "0"

        results.append({
            "question_id": row["question_id"],
            "split": split,
            "type": row["type"],
            "config": cfg.name,
            "fake_mode": use_fake,
            "question": row["question"],
            "answer": result.answer_text,
            "reference_answer": row["reference_answer"],
            "gold_ids": gold_ids,
            "retrieved_ids": retrieved_ids,
            "retrieved_groups": retrieved_groups,
            "retrieval_hit": (
                recall5 > 0 if recall5 is not None else None
            ),
            "recall_at_5": recall5,
            "reciprocal_rank": rr,
            "ndcg_at_10": ndcg10,
            "expected_refusal": expected_refusal,
            "refused": result.refused,
            "refusal_correct": (
                result.refused == expected_refusal
            ),
            "citations": [
                {
                    "label": citation.label,
                    "chunk_id": citation.chunk_id,
                    "covers": citation.covers,
                }
                for citation in result.citations
            ],
            "timings_ms": result.timings_ms,
            "time_ms": result.timings_ms.get("total", 0.0),
        })

    # Calculate aggregate retrieval metrics.
    retrieval_results = [
        item for item in results
        if item["recall_at_5"] is not None
    ]

    expected = [
        item["expected_refusal"] for item in results
    ]
    predicted = [
        item["refused"] for item in results
    ]

    refusals = refusal_metrics(expected, predicted)

    summary = {
        "config": cfg.name,
        "split": split,
        "fake_mode": use_fake,
        "n_questions": len(results),
        "n_retrieval_scored": len(retrieval_results),
        "recall_at_5": (
            mean(item["recall_at_5"] for item in retrieval_results)
            if retrieval_results else None
        ),
        "mrr": (
            mean(item["reciprocal_rank"] for item in retrieval_results)
            if retrieval_results else None
        ),
        "ndcg_at_10": (
            mean(item["ndcg_at_10"] for item in retrieval_results)
            if retrieval_results else None
        ),
        **refusals,
        "average_latency_ms": average_latency([
            item["time_ms"] for item in results
        ]),
    }

    # Create a unique output directory for this run.
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    run_dir = (
        PATHS["root"]
        / "eval"
        / "runs"
        / f"{run_id}_{cfg.name}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    results_file = run_dir / "results.json"
    summary_file = run_dir / "summary.json"

    results_file.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary["run_id"] = run_dir.name

    summary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Append one row to the experiment comparison report.
    reports_dir = PATHS["root"] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_file = reports_dir / f"ablation_{split}.csv"

    fieldnames = [
        "run_id",
        "config",
        "split",
        "fake_mode",
        "n_questions",
        "n_retrieval_scored",
        "recall_at_5",
        "mrr",
        "ndcg_at_10",
        "refusal_precision",
        "refusal_recall",
        "refusal_accuracy",
        "average_latency_ms",
    ]

    write_header = (
        not report_file.exists()
        or report_file.stat().st_size == 0
    )

    with report_file.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )

        if write_header:
            writer.writeheader()

        writer.writerow({
            key: summary[key] for key in fieldnames
        })

    print(f"Evaluated {len(results)} questions")
    print(f"Configuration: {cfg.name}")
    print(f"Fake mode: {use_fake}")
    print(f"Recall@5: {summary['recall_at_5']}")
    print(f"MRR: {summary['mrr']}")
    print(f"nDCG@10: {summary['ndcg_at_10']}")
    print(
        f"Refusal accuracy: "
        f"{summary['refusal_accuracy']:.3f}"
    )
    print(
        f"Average latency: "
        f"{summary['average_latency_ms']:.1f} ms"
    )
    print(f"Results saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"Report updated: {report_file}")

    return results_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the evaluation harness."
    )

    parser.add_argument("--config", required=True)

    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        required=True,
    )

    parser.add_argument(
        "--fake",
        action="store_true",
        help="Use simulated answers instead of the real RAG.",
    )

    args = parser.parse_args()

    evaluate(
        config_path=args.config,
        split=args.split,
        use_fake=args.fake,
    )


if __name__ == "__main__":
    main()
