
"""Run evaluation and save individual and aggregate results.

Usage:
    python -m eval.run_eval --config configs/b1_v1.yaml --split dev --fake

    python -m eval.run_eval --config configs/b1_v1.yaml --split dev --judge

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
    use_judge: bool = False,
    judge_fn=None,
) -> Path:
    """Evaluate questions and save detailed and aggregate results."""

    # Keep the final test set locked.
    if split != "dev":
        raise ValueError(
            "The test set is locked. Only dev is allowed for now."
        )

    # Fake RAG answers must not be judged as real results.
    if use_fake and use_judge:
        raise ValueError(
            "The LLM judge must not score simulated RAG answers."
        )

    cfg = load_config(config_path)

    # Select the answer engine.
    if use_fake:
        from src.generation.fake import answer
    else:
        from src.generation.answer import answer

    # Use a temporary CSV for tests or the official dataset.
    questions_path = (
        Path(questions_path)
        if questions_path is not None
        else PATHS["questions"]
    )

    with questions_path.open(
        newline="", encoding="utf-8"
    ) as file:
        questions = [
            row
            for row in csv.DictReader(file)
            if row["split"].strip() == split
        ]

    if not questions:
        raise ValueError(
            "No dev questions found. Assign dev splits before evaluation."
        )

    results = []

    for row in questions:
        # Follow-up questions can include previous conversation context.
        history = (
            [{"role": "user", "content": row["turn1"]}]
            if row["turn1"].strip()
            else []
        )

        filters = Filters(
            course_code=row["course_code"].strip() or None
        )

        # Generate the answer.
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

        # Retrieval metrics only apply when gold evidence exists.
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

        # Optional LLM-based evaluation.
        judge_result = None

        if use_judge:
            from eval.judge import judge_answer

            sources = [
                {
                    "label": hit.label,
                    "chunk_id": hit.chunk.chunk_id,
                    "covers": hit.chunk.covers,
                    "text": hit.chunk.text,
                }
                for hit in result.context_hits
            ]

            run_judge = (
                judge_fn if judge_fn is not None
                else judge_answer
            )

            judge_result = run_judge(
                answer_text=result.answer_text,
                reference_answer=row["reference_answer"],
                sources=sources,
                cfg=cfg,
            )

        # Save individual question metrics.
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
            "judge": judge_result,
        })

    # Aggregate retrieval metrics.
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

    def metric_mean(items, key):
        values = [
            item[key]
            for item in items
            if item.get(key) is not None
        ]
        return mean(values) if values else None

    summary = {
        "config": cfg.name,
        "split": split,
        "fake_mode": use_fake,
        "n_questions": len(results),
        "n_retrieval_scored": len(retrieval_results),
        "recall_at_5": metric_mean(
            retrieval_results, "recall_at_5"
        ),
        "mrr": metric_mean(
            retrieval_results, "reciprocal_rank"
        ),
        "ndcg_at_10": metric_mean(
            retrieval_results, "ndcg_at_10"
        ),
        **refusals,
        "average_latency_ms": average_latency([
            item["time_ms"] for item in results
        ]),
    }

    # Aggregate LLM judge metrics when available.
    judge_results = [
        item["judge"]
        for item in results
        if item["judge"] is not None
    ]

    summary["n_judged"] = len(judge_results)
    summary["mean_correctness"] = metric_mean(
        judge_results, "correctness"
    )

    total_claims = sum(
        item["n_claims"]
        for item in judge_results
    )

    total_unsupported = sum(
        item["n_unsupported_claims"]
        for item in judge_results
    )

    summary["hallucination_rate"] = (
        total_unsupported / total_claims
        if total_claims else None
    )

    # Save each run in a separate directory.
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

    # Append aggregate metrics to the experiment report.
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
        "n_judged",
        "mean_correctness",
        "hallucination_rate",
    ]

    # Preserve previous reports when adding new columns.
    previous_rows = []

    if report_file.exists() and report_file.stat().st_size:
        with report_file.open(
            newline="", encoding="utf-8"
        ) as file:
            reader = csv.DictReader(file)
            previous_rows = list(reader)

            existing_columns = reader.fieldnames or []

        if existing_columns != fieldnames:
            with report_file.open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(
                    file, fieldnames=fieldnames
                )
                writer.writeheader()

                for previous in previous_rows:
                    writer.writerow({
                        key: previous.get(key, "")
                        for key in fieldnames
                    })

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
            key: summary[key]
            for key in fieldnames
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

    if use_judge:
        print(
            f"Mean correctness: "
            f"{summary['mean_correctness']}"
        )
        print(
            f"Hallucination rate: "
            f"{summary['hallucination_rate']}"
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
        help="Use simulated answers instead of real RAG.",
    )

    parser.add_argument(
        "--judge",
        action="store_true",
        help="Evaluate real answers with the configured LLM judge.",
    )

    args = parser.parse_args()

    evaluate(
        config_path=args.config,
        split=args.split,
        use_fake=args.fake,
        use_judge=args.judge,
    )


if __name__ == "__main__":
    main()
