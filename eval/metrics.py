# Compute the automatic metrics of one evaluation run from its saved answers (results.jsonl, written by
# eval/run_eval.py). No model is called: everything is counted from what the system returned.
#   python -m eval.metrics --run-dir eval/runs/RUN_ID
# Metrics: recall@5 and MRR (did the search find the right source, and how high), refusal accuracy, false
# refusal rate, citation validity and presence, and latency. Whether an answer is correct is graded
# separately by eval/judge.py.
# Writes metrics_questions.csv (one row per question) and metrics_summary.json (the averages) in the run
# folder; eval/error_analysis.py, eval/figures.py and the results of the report read them.

from __future__ import annotations

import argparse
import csv
import json

from pathlib import Path
from statistics import mean


# Read results.jsonl: one JSON record per question, with the question and the answer of the system.
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


# The ids of the original sources (slides, syllabus sections) a retrieved chunk comes from. "covers" holds
# all of them: a chunk of a merged slide can cover several slides.
def source_ids(hit: dict) -> set[str]:

    chunk = hit.get("chunk") or {}

    ids = set(chunk.get("covers") or [])

    if chunk.get("source_id"):
        ids.add(str(chunk["source_id"]))

    if chunk.get("slide_id"):
        ids.add(str(chunk["slide_id"]))

    if chunk.get("section_id"):
        ids.add(str(chunk["section_id"]))

    return ids


# The metrics of one question. The search counts as a hit if any of the first k chunks (the 5 the LLM
# reads) covers at least one gold id; it does not say whether the answer used that source correctly.
# Retrieval metrics are left empty ("") for unanswerable questions, which have no right source.
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

    # Refused is the flag the answer phase sets when it returns the refusal sentence (src/generation/answer.py),
    # not a reading of the answer text.
    refused = bool(
        result.get("refused")
    )

    # Retrieval: the position of the first chunk that covers a gold id (None if none of the first k does).

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

    # A citation is valid when its label ([S1], [S2]...) points to the same chunk the system gave the LLM under
    # that label, so the answer cannot cite a source it did not receive.

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

    # One row of metrics_questions.csv. The refusal decision is correct when the system refused exactly the
    # unanswerable questions; the reciprocal rank is 1/position, or 0 if the right source was not found.

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


# The mean rounded to 4 decimals, or None if there are no values (for example no unanswerable questions).
def average(values: list[float]):

    if not values:
        return None

    return round(
        mean(values),
        4
    )
# The functions from here to average_latency are the general versions of the metrics, checked by
# tests/test_metrics.py; main() below uses score_question instead.

# Turn each retrieved item (one id or a group of ids) into a set of ids.
def _source_sets(retrieved):
    return [
        {source} if isinstance(source, str) else set(source)
        for source in retrieved
    ]


# Share of the gold ids found in the first k results. Unlike score_question (found or not), it gives
# partial credit when only some of the gold ids are found.
def recall_at_k(retrieved, gold, k=5):
    if k < 1:
        return 0.0

    gold_set = set(gold)

    if not gold_set:
        return 0.0

    retrieved_ids = set()

    for source in _source_sets(retrieved)[:k]:
        retrieved_ids.update(source)

    return len(retrieved_ids & gold_set) / len(gold_set)


# 1 divided by the position of the first result that covers a gold id, or 0 if none does.
def reciprocal_rank(retrieved, gold):
    gold_set = set(gold)

    if not gold_set:
        return 0.0

    for rank, source in enumerate(_source_sets(retrieved), start=1):
        if source & gold_set:
            return 1.0 / rank

    return 0.0


# nDCG@k: every right result found adds more the higher it is ranked (1/log2(position + 1)), divided by the
# best score possible, so 1 means all the right sources are at the top.
def ndcg_at_k(retrieved, gold, k=5):
    from math import log2

    if k < 1:
        return 0.0

    gold_set = set(gold)

    if not gold_set:
        return 0.0

    sources = _source_sets(retrieved)[:k]

    dcg = sum(
        1.0 / log2(rank + 1)
        for rank, source in enumerate(sources, start=1)
        if source & gold_set
    )

    ideal_count = min(len(gold_set), k)

    idcg = sum(
        1.0 / log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    return dcg / idcg if idcg else 0.0


# Compare the refusals expected with the refusals made: precision (refusals that were right), recall
# (questions to refuse that were refused) and accuracy. Each ratio is 0 when its denominator is 0.
def refusal_metrics(expected, predicted):
    if len(expected) != len(predicted):
        raise ValueError("Expected and predicted lengths must match")

    if not expected:
        return {
            "refusal_precision": 0.0,
            "refusal_recall": 0.0,
            "refusal_accuracy": 0.0,
        }

    tp = sum(
        bool(e) and bool(p)
        for e, p in zip(expected, predicted)
    )

    fp = sum(
        not bool(e) and bool(p)
        for e, p in zip(expected, predicted)
    )

    fn = sum(
        bool(e) and not bool(p)
        for e, p in zip(expected, predicted)
    )

    correct = sum(
        bool(e) == bool(p)
        for e, p in zip(expected, predicted)
    )

    return {
        "refusal_precision": tp / (tp + fp) if tp + fp else 0.0,
        "refusal_recall": tp / (tp + fn) if tp + fn else 0.0,
        "refusal_accuracy": correct / len(expected),
    }


# Average time per answer, in milliseconds.
def average_latency(latencies):
    return sum(latencies) / len(latencies) if latencies else 0.0

# Score every question of a run, average the metrics and save both files in the run folder.
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

    # Load the answers. Questions that failed with an error (status not "ok") are counted but not scored.

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

    # Score each question, then split them into answerable and unanswerable for the averages.

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

    # The averages. Recall and MRR use only the answerable questions; the false refusal rate is the share of
    # answerable questions refused, and correct_unanswerable_refusal_rate the share of unanswerable ones
    # refused. Citation validity is valid citations over all citations of the run.

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

    # Save the metrics of each question (read by eval/error_analysis.py).

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

    # Save the averages (read by eval/figures.py).

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

    # Print the averages.

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


# Runs only when the file is executed directly.
if __name__ == "__main__":
    main()
