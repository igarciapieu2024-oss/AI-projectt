"""
Run reproducible RAG evaluation on dev or the held-out test split.

Each run records answers and evidence in JSONL plus a summary JSON. The CLI
requires explicit ``--allow-test`` for held-out questions and saves each
question immediately so partial failures can be diagnosed. The separate
``evaluate`` function supports isolated fake-engine unit tests.

Usage:
python -m eval.run_eval --config configs/final_system.yaml --split dev --limit 2

Owner: Elena.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src.config import load_config, PATHS
from src.generation.answer import answer


def make_jsonable(value):
    """Convert nested engine results to safe JSON-compatible structures.

    Non-finite floating-point values become null because standard JSON
    cannot represent NaN or infinity reliably.
    """

    if hasattr(value, "__dataclass_fields__"):
        return make_jsonable(asdict(value))

    if isinstance(value, dict):
        return {
            str(key): make_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def read_questions(path: Path, split: str):
    """Select only the requested split from the question bank CSV.

    ``utf-8-sig`` accepts files exported with a UTF-8 byte-order mark.
    """

    with path.open(
        encoding="utf-8-sig",
        newline=""
    ) as file:

        rows = list(csv.DictReader(file))

    selected = [
        row
        for row in rows
        if row.get("split", "").strip() == split
    ]

    if not selected:
        raise ValueError(
            f"No questions assigned to split={split}"
        )

    return selected


def save_jsonl(path: Path, record: dict):
    """Append one result immediately to preserve progress if a run fails."""

    with path.open("a", encoding="utf-8") as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=str
            ) + "\n"
        )
def evaluate(
    config_path="configs/final_system.yaml",
    split="dev",
    use_fake=False,
    questions_path=None,
    use_judge=False,
    judge_fn=None,
):
    """Evaluation interface supporting deterministic fake-mode tests.

    This interface is separate from the production CLI runner and always
    blocks the test split. Production's authorized test runs use ``main``.
    """

    if split == "test":
        raise ValueError("The test split is locked")

    if split != "dev":
        raise ValueError(f"Unsupported split: {split}")

    cfg = load_config(config_path)

    if questions_path is None:
        questions_path = PATHS["root"] / "eval" / "questions.csv"

    questions_path = Path(questions_path)

    selected = read_questions(questions_path, split)

    if use_fake:
        from src.generation.fake import answer as answer_fn
    else:
        from importlib import import_module
        answer_fn = import_module(
            "src.generation.answer"
        ).answer

    if use_judge and judge_fn is None:
        from eval.judge import judge_answer
        judge_fn = judge_answer

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

    output_path = run_dir / "results.json"

    results = []

    for row in selected:
        question = row["question"]

        history = None
        first_turn = (row.get("turn1") or "").strip()

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

        result = answer_fn(
            question,
            history=history,
            cfg=cfg,
        )

        gold = {
            item.strip()
            for item in row.get("gold_ids", "").split(";")
            if item.strip()
        }

        retrieved = [
            set(hit.chunk.covers)
            for hit in result.context_hits
        ]

        found = set()

        for source in retrieved[:5]:
            found.update(source)

        answerable = row["answerable"] == "1"

        retrieval_hit = bool(gold & found)

        recall = (
            len(gold & found) / len(gold)
            if gold
            else None
        )

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

    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

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

    def mean_or_none(values):
        return sum(values) / len(values) if values else None

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

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    reports_dir = PATHS["root"] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / f"ablation_{split}.csv"

    report_columns = list(summary.keys())

    previous_rows = []

    if report_path.exists():
        with report_path.open(
            encoding="utf-8",
            newline="",
        ) as f:
            previous_rows = list(csv.DictReader(f))

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

    return output_path

def main():
    """Run the selected config and split, persisting each question result.

    The protected test split requires an explicit command-line opt-in.
    Follow-up questions first run their initial turn and pass its answer as
    conversation history to the evaluated second turn.
    """

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

    # Protect the test set

    if args.split == "test" and not args.allow_test:

        parser.error(
            "Test split is protected. "
            "Use --allow-test for the final evaluation."
        )

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

    if args.limit is not None:
        selected = selected[:args.limit]

    # Create run directory

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

    results_path = folder / "results.jsonl"

    started = time.perf_counter()

    successes = 0
    failures = 0

    latencies = []

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

        qid = row["question_id"]

        question = row["question"]

        print(
            f"[{number}/{len(selected)}] "
            f"{qid}: {question[:70]}",
            flush=True
        )

        try:

            history = None

            first_turn = (
                row.get("turn1") or ""
            ).strip()

            first_turn_result = None

            # Handle follow-up questions

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

            # Save evaluation information

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

            latency = float(
                result.timings_ms.get("total", 0)
            )

            latencies.append(latency)

            successes += 1

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

        # Write each result immediately

        save_jsonl(
            results_path,
            payload
        )

    # Calculate execution summary

    if latencies:

        mean_latency = round(
            sum(latencies) / len(latencies),
            1
        )

    else:

        mean_latency = None

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

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
