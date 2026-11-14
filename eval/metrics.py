"""
Compute evaluation metrics from saved RAG results.

Metrics:
- Recall@k
- MRR@k
- Refusal accuracy
- False refusal rate
- Citation validity
- Citation presence
- Latency

Owner: Elena.
"""

from __future__ import annotations

import argparse
import csv
import json

from pathlib import Path
from statistics import mean


def load_run(path: Path) -> list[dict]:

    if not path.is_file():
        raise FileNotFoundError(
            f"Results file not found: {path}"
        )

    with path.open(encoding="utf-8") as file:

        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def source_ids(hit: dict) -> set[str]:
    """Identify the original sources covered by a chunk."""

    chunk = hit.get("chunk") or {}

    ids = set(chunk.get("covers") or [])

    if chunk.get("source_id"):
        ids.add(str(chunk["source_id"]))

    if chunk.get("slide_id"):
        ids.add(str(chunk["slide_id"]))

    if chunk.get("section_id"):
        ids.add(str(chunk["section_id"]))

    return ids


def score_question(row: dict, k: int) -> dict:

    result = row["result"]

    hits = result.get("context_hits") or []

    citations = result.get("citations") or []

    gold = {
        item.strip()
        for item in str(
            row.get("gold_ids") or ""
        ).split(";")
        if item.strip()
    }

    answerable = (
        str(row.get("answerable", "")).strip() == "1"
    )

    refused = bool(
        result.get("refused")
    )

    # Retrieval metrics

    first_rank = None

    if answerable:

        for rank, hit in enumerate(
            hits[:k],
            start=1
        ):

            retrieved_ids = source_ids(hit)

            if gold.intersection(retrieved_ids):

                first_rank = rank

                break

    # Citation validation

    labels = {
        hit.get("label"): (
            hit.get("chunk") or {}
        ).get("chunk_id")
        for hit in hits
    }

    valid_citations = sum(
        citation.get("label") in labels
        and labels[citation.get("label")]
        == citation.get("chunk_id")
        for citation in citations
    )

    total_citations = len(citations)

    # Build question-level metrics

    return {
        "question_id": row.get("question_id", ""),
        "type": row.get("type", ""),
        "answerable": int(answerable),
        "refused": int(refused),

        "correct_refusal_decision": int(
            refused != answerable
        ),

        "recall_at_k": (
            int(first_rank is not None)
            if answerable
            else ""
        ),

        "reciprocal_rank": (
            round(1 / first_rank, 6)
            if first_rank
            else (0.0 if answerable else "")
        ),

        "retrieved_count": len(hits),

        "citation_count": total_citations,

        "valid_citation_count": valid_citations,

        "citation_present": int(
            total_citations > 0
        ),

        "citation_precision": (
            round(
                valid_citations / total_citations,
                6
            )
            if total_citations
            else ""
        ),

        "latency_ms": (
            result.get("timings_ms") or {}
        ).get("total", "")
    }


def average(values: list[float]):

    if not values:
        return None

    return round(
        mean(values),
        4
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5
    )

    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be at least 1")

    # Load results

    results_path = (
        args.run_dir / "results.jsonl"
    )

    rows = load_run(results_path)

    successful = [
        row
        for row in rows
        if row.get("status") == "ok"
    ]

    failed = [
        row
        for row in rows
        if row.get("status") != "ok"
    ]

    if not successful:
        parser.error(
            "No successful question results to score"
        )

    # Score each question

    scores = [
        score_question(row, args.k)
        for row in successful
    ]

    answerable = [
        row
        for row in scores
        if row["answerable"] == 1
    ]

    unanswerable = [
        row
        for row in scores
        if row["answerable"] == 0
    ]

    all_citations = sum(
        row["citation_count"]
        for row in scores
    )

    valid_citations = sum(
        row["valid_citation_count"]
        for row in scores
    )

    # Aggregate metrics

    summary = {
        "run_dir": str(args.run_dir),

        "evaluated": len(scores),

        "failed": len(failed),

        "answerable": len(answerable),

        "unanswerable": len(unanswerable),

        "recall_at_k": average([
            row["recall_at_k"]
            for row in answerable
        ]),

        "mrr_at_k": average([
            row["reciprocal_rank"]
            for row in answerable
        ]),

        "refusal_accuracy": average([
            row["correct_refusal_decision"]
            for row in scores
        ]),

        "false_refusal_rate": average([
            row["refused"]
            for row in answerable
        ]),

        "correct_unanswerable_refusal_rate": average([
            row["refused"]
            for row in unanswerable
        ]),

        "citation_presence_rate": average([
            row["citation_present"]
            for row in scores
        ]),

        "citation_validity": (
            round(
                valid_citations / all_citations,
                4
            )
            if all_citations
            else None
        ),

        "mean_latency_ms": average([
            float(row["latency_ms"])
            for row in scores
            if row["latency_ms"] != ""
        ]),

        "notes": [
            "Recall and MRR check whether a top-k chunk covers a gold source ID.",
            "Citation validity checks cited labels and chunk IDs.",
            "Citation validity is null if no citations were generated.",
            "Answer correctness requires a separate judge."
        ]
    }

    # Save question-level metrics

    csv_path = (
        args.run_dir / "metrics_questions.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(scores[0].keys())
        )

        writer.writeheader()

        writer.writerows(scores)

    # Save aggregate metrics

    json_path = (
        args.run_dir / "metrics_summary.json"
    )

    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        ) + "\n",
        encoding="utf-8"
    )

    # Display results

    print(
        "\n========== EVALUATION METRICS =========="
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        )
    )

    print(
        "========================================"
    )

    print(f"\nSaved: {csv_path}")

    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()