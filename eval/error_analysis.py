"""Analyze evaluation errors and assign one primary cause per failed question.

Usage:
    python -m eval.error_analysis --run-dir eval/runs/RUN_ID

Output:
    reports/error_analysis.csv

Owner: Elena.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    """Read a CSV file into a list of dictionaries."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_bool(value) -> bool:
    """Convert CSV boolean values."""
    return str(value).strip().lower() in {"1", "true", "yes"}


def classify_error(judge: dict, metrics: dict) -> tuple[str, str]:
    """Assign one primary error category using available automatic signals.

    The ordering matters: refusal mistakes take precedence over retrieval,
    missing citations, and generation errors. A generation error label only
    means gold evidence was available; it is not proof of a model mistake.
    All such cases should be checked against the original answer manually.
    """

    answerable = to_bool(metrics.get("answerable"))
    refused = to_bool(metrics.get("refused"))
    recall = to_bool(metrics.get("recall_at_k"))
    citation_present = to_bool(metrics.get("citation_present"))

    if answerable and refused:
        return (
            "false_refusal",
            "The system refused an answerable question."
        )

    if not answerable and not refused:
        return (
            "incorrect_non_refusal",
            "The system answered a question that should have been refused."
        )

    if answerable and not recall:
        return (
            "retrieval_error",
            "No retrieved chunk covered the annotated gold source IDs."
        )

    if answerable and not citation_present:
        return (
            "citation_error",
            "The answer did not contain a valid source citation."
        )

    if answerable and recall and citation_present:
        return (
            "generation_error",
            "Gold evidence was retrieved and cited, but the judge marked the answer incorrect. Requires manual verification."
        )

    return (
        "needs_review",
        "The available metrics do not establish a single error cause."
    )


def main() -> None:
    """Join per-question judge and metric files into a reviewable error CSV."""

    parser = argparse.ArgumentParser(
        description="Analyze RAG evaluation errors."
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        help="Directory containing evaluation results."
    )

    parser.add_argument(
        "--questions",
        default="eval/questions.csv",
        help="Path to the evaluation question bank."
    )

    parser.add_argument(
        "--output",
        default="reports/error_analysis.csv",
        help="Output CSV path."
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    questions_path = Path(args.questions)
    output_path = Path(args.output)

    judge_path = run_dir / "judge_questions.csv"
    metrics_path = run_dir / "metrics_questions.csv"

    for path in [judge_path, metrics_path, questions_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    judges = read_csv(judge_path)
    metrics = read_csv(metrics_path)
    questions = read_csv(questions_path)

    metrics_by_id = {
        row["question_id"]: row
        for row in metrics
    }

    questions_by_id = {
        row["question_id"]: row
        for row in questions
    }

    errors = []

    for judge in judges:

        qid = judge["question_id"]

        if judge.get("status") != "ok":
            errors.append({
                "question_id": qid,
                "type": questions_by_id.get(qid, {}).get("type", ""),
                "course_code": questions_by_id.get(qid, {}).get("course_code", ""),
                "question": questions_by_id.get(qid, {}).get("question", ""),
                "reference_answer": questions_by_id.get(qid, {}).get("reference_answer", ""),
                "judge_score": judge.get("score", ""),
                "judge_reason": judge.get("reason", ""),
                "error_type": "judge_error",
                "explanation": "The correctness judge did not evaluate this question successfully.",
                "manual_review": "yes",
            })
            continue

        if to_bool(judge.get("correct")):
            continue

        metric = metrics_by_id.get(qid)
        question = questions_by_id.get(qid)

        if metric is None or question is None:
            raise ValueError(f"Missing metrics or question for {qid}")

        error_type, explanation = classify_error(judge, metric)

        errors.append({
            "question_id": qid,
            "type": question.get("type", ""),
            "course_code": question.get("course_code", ""),
            "question": question.get("question", ""),
            "reference_answer": question.get("reference_answer", ""),
            "judge_score": judge.get("score", ""),
            "judge_reason": judge.get("reason", ""),
            "error_type": error_type,
            "explanation": explanation,
            "manual_review": "yes",
        })

    columns = [
        "question_id",
        "type",
        "course_code",
        "question",
        "reference_answer",
        "judge_score",
        "judge_reason",
        "error_type",
        "explanation",
        "manual_review",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(errors)

    counts = Counter(row["error_type"] for row in errors)

    print("\n========== ERROR ANALYSIS ==========")
    print("Run:", run_dir)
    print("Questions judged:", len(judges))
    print("Incorrect or failed evaluations:", len(errors))

    print("\nErrors by category:")
    for category, count in sorted(counts.items()):
        print(f"  {category}: {count}")

    print("\nSaved:", output_path)
    print("====================================")


if __name__ == "__main__":
    main()
