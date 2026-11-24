# Run the system on a set of evaluation questions and save every answer with its sources, so the metrics
# (eval/metrics.py) and the judge (eval/judge.py) can be calculated afterwards without asking the LLM again.
#   python -m eval.run_eval --config configs/final_system.yaml --split dev --limit 2
#   python -m eval.run_eval --config configs/final_system.yaml --questions eval/questions_holdout3.csv --split test
# --allow-test
# Each run gets its own folder eval/runs/<time>_<config>_<split>/ with results.jsonl (one record per question) and
# summary.json. The test questions need --allow-test, so we don't run them by mistake while the system is still being
# changed. main() is the real evaluation; evaluate() is a smaller version used by the tests.

from __future__ import annotations

# argparse reads the options, csv reads the questions, json writes the results, math checks the numbers and time
# measures the run.
import argparse
import csv
import json
import math
import time

# asdict turns a result into a dictionary, and datetime gives the date and time of the run for the folder name.
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# The config reader, the folders of the project and the answer phase itself.
from src.config import load_config, PATHS
from src.generation.answer import answer


# Turn the result of the answer phase (dataclasses with lists, paths and numbers inside) into plain dictionaries and
# lists that can be written as JSON. NaN or infinite numbers become null, because JSON can't store them.
def make_jsonable(value):

    # A record (dataclass): turned into a dictionary first, and then its values are converted too.
    if hasattr(value, "__dataclass_fields__"):
        return make_jsonable(asdict(value))

    # A dictionary: its keys become text and its values are converted.
    if isinstance(value, dict):
        return {
            str(key): make_jsonable(item)
            for key, item in value.items()
        }

    # A list or a tuple: each item is converted.
    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]

    # A path becomes text.
    if isinstance(value, Path):
        return str(value)

    # A number that isn't finite (NaN or infinity) becomes None, which JSON writes as null.
    if isinstance(value, float) and not math.isfinite(value):
        return None

    # Anything else (text, numbers, true/false) is already fine.
    return value


# Read the question file and keep only the questions of the requested split (dev or test). utf-8-sig also accepts
# files saved by Excel.
def read_questions(path: Path, split: str):

    # Read every row of the file.
    with path.open(
        encoding="utf-8-sig",
        newline=""
    ) as file:

        rows = list(csv.DictReader(file))

    # Keep the rows of the requested split.
    selected = [
        row
        for row in rows
        if row.get("split", "").strip() == split
    ]

    # No question in that split is a mistake in the command.
    if not selected:
        raise ValueError(
            f"No questions assigned to split={split}"
        )

    return selected


# Add one result at the end of results.jsonl as soon as it's ready, so the answers already given are kept if the run
# stops halfway.
def save_jsonl(path: Path, record: dict):

    # Mode "a" adds to the end of the file; default=str writes anything JSON doesn't know as text.
    with path.open("a", encoding="utf-8") as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=str
            ) + "\n"
        )
# A smaller evaluation used by tests/test_run_eval.py. With use_fake it answers with src/generation/fake.py (no
# models), so the tests are fast and always give the same result. It only accepts the dev split; the real runs,
# including the test questions, use main() below.
def evaluate(
    config_path="configs/final_system.yaml",
    split="dev",
    use_fake=False,
    questions_path=None,
    use_judge=False,
    judge_fn=None,
):

    # This small version only runs the dev questions.
    if split == "test":
        raise ValueError("The test split is locked")

    if split != "dev":
        raise ValueError(f"Unsupported split: {split}")

    # The configuration, and the question file (eval/questions.csv by default).
    cfg = load_config(config_path)

    if questions_path is None:
        questions_path = PATHS["root"] / "eval" / "questions.csv"

    questions_path = Path(questions_path)

    selected = read_questions(questions_path, split)

    # The answer function: the fake one for the tests, or the real answer phase.
    if use_fake:
        from src.generation.fake import answer as answer_fn
    else:
        from importlib import import_module
        answer_fn = import_module(
            "src.generation.answer"
        ).answer

    # The judge is only loaded when it is asked for.
    if use_judge and judge_fn is None:
        from eval.judge import judge_answer
        judge_fn = judge_answer

    # The folder of the run, named with the time, the config and the split.
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    run_dir = (
        PATHS["root"]
        / "eval"
        / "runs"
        / f"{timestamp}_{cfg.name}_{split}"
    )

    run_dir.mkdir(parents=True, exist_ok=False)

    # This small version saves all the results together, in one JSON file.
    output_path = run_dir / "results.json"

    results = []

    # Go through the questions.
    for row in selected:
        question = row["question"]

        # No conversation unless the question is a follow-up.
        history = None
        first_turn = (row.get("turn1") or "").strip()

        # A follow-up question: we answer the first question first, and pass both as the conversation.
        if first_turn:
            first_result = answer_fn(first_turn, cfg=cfg)

            history = [
                {
                    "role": "user",
                    "content": first_turn,
                },
                {
                    "role": "assistant",
                    "content": first_result.answer_text,
                },
            ]

        # The answer to the question itself.
        result = answer_fn(
            question,
            history=history,
            cfg=cfg,
        )

        # The gold ids of the question.
        gold = {
            item.strip()
            for item in row.get("gold_ids", "").split(";")
            if item.strip()
        }

        # The sources covered by the first 5 chunks, and the share of the gold ids among them.
        retrieved = [
            set(hit.chunk.covers)
            for hit in result.context_hits
        ]

        found = set()

        for source in retrieved[:5]:
            found.update(source)

        # Whether the question can be answered, and whether the search found a gold source.
        answerable = row["answerable"] == "1"

        retrieval_hit = bool(gold & found)

        recall = (
            len(gold & found) / len(gold)
            if gold
            else None
        )

        # The record of this question.
        record = {
            "question_id": row["question_id"],
            "question": question,
            "answer_text": result.answer_text,
            "retrieval_hit": retrieval_hit,
            "recall_at_5": recall,
            "refused": bool(result.refused),
            "refusal_correct": (
                bool(result.refused) != answerable
            ),
            "citations": make_jsonable(result.citations),
            "latency_ms": result.timings_ms.get("total", 0),
        }

        # Optionally we grade the answer with the judge, giving it the sources the system used.
        if use_judge:
            sources = [
                {
                    "label": hit.label,
                    "text": hit.chunk.text,
                    "covers": hit.chunk.covers,
                }
                for hit in result.context_hits
            ]

            record["judge"] = judge_fn(
                answer_text=result.answer_text,
                reference_answer=row.get("reference_answer", ""),
                sources=sources,
                cfg=cfg,
            )

        results.append(record)

    # Write all the records.
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # The averages of the run, saved in summary.json and added as a row to reports/ablation_<split>.csv.
    retrieval_scores = [
        row["recall_at_5"]
        for row in results
        if row["recall_at_5"] is not None
    ]

    judged = [
        row["judge"]
        for row in results
        if "judge" in row
    ]

    # The mean of a list, or None when the list is empty.
    def mean_or_none(values):
        return sum(values) / len(values) if values else None

    # The hallucination rates the judge gave, when there are any.
    hallucination_scores = [
        row["hallucination_rate"]
        for row in judged
        if row.get("hallucination_rate") is not None
    ]

    summary = {
        "config": cfg.name,
        "split": split,
        "fake_mode": bool(use_fake),
        "n_questions": len(results),
        "n_retrieval_scored": len(retrieval_scores),
        "recall_at_5": mean_or_none(retrieval_scores),
        "refusal_accuracy": mean_or_none([
            float(row["refusal_correct"])
            for row in results
        ]),
        "n_judged": len(judged),
        "mean_correctness": mean_or_none([
            row["correctness"]
            for row in judged
        ]),
        "hallucination_rate": mean_or_none(
            hallucination_scores
        ),
    }

    # Save the summary in the folder of the run.
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # The report file in reports/, with one row per run.
    reports_dir = PATHS["root"] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / f"ablation_{split}.csv"

    report_columns = list(summary.keys())

    previous_rows = []

    # The rows already in the report are kept, and the new run is added at the end.
    if report_path.exists():
        with report_path.open(
            encoding="utf-8",
            newline="",
        ) as f:
            previous_rows = list(csv.DictReader(f))

    # Write the report again: the header, the old rows and the new one.
    with report_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=report_columns,
        )

        writer.writeheader()

        for previous in previous_rows:
            writer.writerow({
                key: previous.get(key, "")
                for key in report_columns
            })

        writer.writerow(summary)

    # The path of the results file.
    return output_path

# The real evaluation: run one config on one set of questions and save each answer as soon as it's ready. A follow-up
# question is run in two turns: first its first question, and then the follow-up with the first answer as the
# conversation, as in the app.
def main():

    # The options of the command.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/final_system.yaml"
    )

    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev"
    )

    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("eval/questions.csv")
    )

    # --limit runs only the first questions, to check a change quickly.
    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/runs")
    )

    parser.add_argument(
        "--allow-test",
        action="store_true"
    )

    args = parser.parse_args()

    # Protect the test set: the test questions are only run on purpose, once the system is fixed.

    if args.split == "test" and not args.allow_test:

        parser.error(
            "Test split is protected. "
            "Use --allow-test for the final evaluation."
        )

    # --limit has to be at least 1.
    if args.limit is not None and args.limit < 1:

        parser.error(
            "--limit must be a positive number"
        )

    # Load configuration

    cfg = load_config(args.config)

    # Load questions

    selected = read_questions(
        args.questions,
        args.split
    )

    # With --limit, only the first questions.
    if args.limit is not None:
        selected = selected[:args.limit]

    # Create the folder of the run: the time, the config and the split, so no run overwrites another one.

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    run_id = (
        f"{timestamp}_{cfg.name}_{args.split}"
    )

    folder = args.output_dir / run_id

    folder.mkdir(
        parents=True,
        exist_ok=False
    )

    # The file where each answer is saved.
    results_path = folder / "results.jsonl"

    # The start time, and the counters of the run.
    started = time.perf_counter()

    successes = 0
    failures = 0

    latencies = []

    # Print what is about to run.
    print(
        f"Run: {run_id}"
    )

    print(
        f"Config: {cfg.name}"
    )

    print(
        f"Split: {args.split}"
    )

    print(
        f"Questions: {len(selected)}"
    )

    # Evaluate every question

    for number, row in enumerate(selected, start=1):

        # The id and the text of the question; print it so we can follow the run.
        qid = row["question_id"]

        question = row["question"]

        print(
            f"[{number}/{len(selected)}] "
            f"{qid}: {question[:70]}",
            flush=True
        )

        try:

            # No conversation unless the question is a follow-up.
            history = None

            first_turn = (
                row.get("turn1") or ""
            ).strip()

            first_turn_result = None

            # Follow-up questions: we answer the first question and keep it as the conversation.

            if first_turn:

                first = answer(
                    first_turn,
                    cfg=cfg
                )

                first_turn_result = make_jsonable(
                    first
                )

                history = [
                    {
                        "role": "user",
                        "content": first_turn
                    },
                    {
                        "role": "assistant",
                        "content": first.answer_text
                    }
                ]

            # Generate the main answer

            result = answer(
                question,
                history=history,
                cfg=cfg
            )

            # Everything the metrics and the judge need: the question with its gold ids and its reference answer, and
            # the full result of the system (the answer, the citations, the chunks it used and the times).

            payload = {
                "question_id": qid,
                "split": args.split,
                "type": row["type"],
                "author": row["author"],
                "question": question,
                "turn1": first_turn,
                "gold_ids": row.get(
                    "gold_ids", ""
                ),
                "reference_answer": row.get(
                    "reference_answer", ""
                ),
                "key_facts": row.get(
                    "key_facts", ""
                ),
                "answerable": row.get(
                    "answerable", ""
                ),
                "config": cfg.name,
                "status": "ok",
                "first_turn_result": first_turn_result,
                "result": make_jsonable(result)
            }

            # The time this answer took.
            latency = float(
                result.timings_ms.get("total", 0)
            )

            latencies.append(latency)

            successes += 1

        # A question that fails is saved with its error, and the run goes on with the next one.
        except Exception as exc:

            failures += 1

            payload = {
                "question_id": qid,
                "split": args.split,
                "config": cfg.name,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc)
            }

            print(
                f"ERROR: {type(exc).__name__}: {exc}",
                flush=True
            )

        # Write each result straight away.

        save_jsonl(
            results_path,
            payload
        )

    # The summary of the run. metrics_status "pending" means eval/metrics.py hasn't been run on it yet.

    # The average time per answer (None if no question worked).
    if latencies:

        mean_latency = round(
            sum(latencies) / len(latencies),
            1
        )

    else:

        mean_latency = None

    # The summary of the run.
    summary = {
        "run_id": run_id,
        "config": cfg.name,
        "config_file": args.config,
        "split": args.split,
        "kb_sig": cfg.kb_sig,
        "requested": len(selected),
        "completed": successes,
        "failed": failures,
        "mean_latency_ms": mean_latency,
        "wall_time_s": round(
            time.perf_counter() - started,
            2
        ),
        "results_file": str(results_path),
        "metrics_status": "pending"
    }

    # Save it as summary.json and print it.
    summary_path = folder / "summary.json"

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        ) + "\n",
        encoding="utf-8"
    )

    print("\n========== EVALUATION SUMMARY ==========")

    print(
        json.dumps(
            summary,
            indent=2
        )
    )

    print("========================================")

    # Leave with an error code if any question failed, so it isn't missed.
    if failures:
        raise SystemExit(1)


# Runs only when the file is executed directly.
if __name__ == "__main__":
    main()
