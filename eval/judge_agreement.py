"""
Agreement between a judge and reference grades made by checking each answer against its sources.

1. The reference grades in eval/reference_labels_dev.csv were made blind (answers of two configurations
   shuffled together) by a second model that read the gold slide or syllabus section of every question.
   They are not human grades; a sample is checked by hand and reported with them.
2. Only dev runs are graded, so the judge can be chosen without looking at any test question.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_scores(path: Path) -> dict[str, float]:
    """3. question_id -> score of one judge file; rows the judge could not score are skipped."""
    scores = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status", "ok") == "ok" and row.get("score", "") != "":
                scores[row["question_id"]] = float(row["score"])
    return scores


def agreement(reference: dict[tuple[str, str], float], judged: dict[tuple[str, str], float]) -> dict:
    """4. Same score, same correct/not-correct decision, and how often each side calls an answer correct."""
    keys = [key for key in reference if key in judged]
    if not keys:
        return {"compared": 0}
    same = sum(reference[k] == judged[k] for k in keys)
    same_decision = sum((reference[k] == 1) == (judged[k] == 1) for k in keys)
    too_strict = sum(reference[k] == 1 and judged[k] < 1 for k in keys)
    too_lenient = sum(reference[k] < 1 and judged[k] == 1 for k in keys)
    return {
        "compared": len(keys),
        "same_score": round(same / len(keys), 4),
        "same_correct_decision": round(same_decision / len(keys), 4),
        "judge_too_strict": too_strict,
        "judge_too_lenient": too_lenient,
        "reference_exact_correct_rate": round(sum(reference[k] == 1 for k in keys) / len(keys), 4),
        "judge_exact_correct_rate": round(sum(judged[k] == 1 for k in keys) / len(keys), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("eval/reference_labels_dev.csv"))
    parser.add_argument("--runs-dir", type=Path, default=Path("eval/runs"))
    args = parser.parse_args()

    reference: dict[tuple[str, str], float] = {}
    with args.labels.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            reference[(row["run_id"], row["question_id"])] = float(row["reference_score"])

    run_ids = sorted({run_id for run_id, _ in reference})
    for name in ("judge_questions.csv", "judge_questions_v2.csv", "judge_questions_v3.csv"):
        judged = {}
        for run_id in run_ids:
            path = args.runs_dir / run_id / name
            if path.exists():
                judged.update({(run_id, qid): score for qid, score in load_scores(path).items()})
        print(name, agreement(reference, judged))


if __name__ == "__main__":
    main()
