# Measure how often the automatic judge (eval/judge.py) agrees with the reference grades, which were made by checking
# each answer against the source of its question.
# The reference grades in eval/reference_labels_dev.csv were made blind (the answers of two configurations shuffled
# together) by a second model that read the gold slide or syllabus section of every question. They aren't human
# grades; a sample is checked by hand and reported with them.
# We only compare dev runs, so the judge can be chosen without looking at any test question.
#   python -m eval.judge_agreement

from __future__ import annotations

# argparse reads the options of the command, and csv reads the grade files.
import argparse
import csv
from pathlib import Path


# Read one judge file: question_id -> score. We skip the rows the judge couldn't score (status not "ok").
def load_scores(path: Path) -> dict[str, float]:
    # question_id -> score.
    scores = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status", "ok") == "ok" and row.get("score", "") != "":
                scores[row["question_id"]] = float(row["score"])
    return scores


# Compare the judge with the reference on the answers both graded: how often they give the same score, how often they
# agree on "fully correct or not", how many correct answers the judge marked lower (too strict) or wrong answers it
# marked correct (too lenient), and the share of fully correct answers for each side.
def agreement(reference: dict[tuple[str, str], float], judged: dict[tuple[str, str], float]) -> dict:
    # The answers that both the reference and the judge graded.
    keys = [key for key in reference if key in judged]
    # With nothing in common there is nothing to compare.
    if not keys:
        return {"compared": 0}
    # The four counts: the same score, the same "fully correct or not" decision, the judge stricter and the judge more
    # lenient.
    same = sum(reference[k] == judged[k] for k in keys)
    same_decision = sum((reference[k] == 1) == (judged[k] == 1) for k in keys)
    too_strict = sum(reference[k] == 1 and judged[k] < 1 for k in keys)
    too_lenient = sum(reference[k] < 1 and judged[k] == 1 for k in keys)
    # The counts, and the shares rounded to 4 decimals.
    return {
        "compared": len(keys),
        "same_score": round(same / len(keys), 4),
        "same_correct_decision": round(same_decision / len(keys), 4),
        "judge_too_strict": too_strict,
        "judge_too_lenient": too_lenient,
        "reference_exact_correct_rate": round(sum(reference[k] == 1 for k in keys) / len(keys), 4),
        "judge_exact_correct_rate": round(sum(judged[k] == 1 for k in keys) / len(keys), 4),
    }


# Load the reference grades, and then compare each version of the judge (the v1, v2 and v3 files) with them.
def main() -> None:
    # The two options: the file with the reference grades and the folder with the runs.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("eval/reference_labels_dev.csv"))
    parser.add_argument("--runs-dir", type=Path, default=Path("eval/runs"))
    args = parser.parse_args()

    # (run_id, question_id) -> reference score, so the same question in two runs is kept apart.
    reference: dict[tuple[str, str], float] = {}
    with args.labels.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            reference[(row["run_id"], row["question_id"])] = float(row["reference_score"])

    # Only the runs that have reference grades.
    run_ids = sorted({run_id for run_id, _ in reference})
    # The three versions of the judge's rubric, each one saved in its own file inside the folder of the run.
    for name in ("judge_questions.csv", "judge_questions_v2.csv", "judge_questions_v3.csv"):
        judged = {}
        for run_id in run_ids:
            # Add the grades of this run, if the judge graded it with this rubric.
            path = args.runs_dir / run_id / name
            if path.exists():
                judged.update({(run_id, qid): score for qid, score in load_scores(path).items()})
        # Print the agreement of this rubric with the reference grades.
        print(name, agreement(reference, judged))


# Runs only when the file is executed directly.
if __name__ == "__main__":
    main()
