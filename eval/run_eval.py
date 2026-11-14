"""
Run RAG evaluation on dev or test questions.

Usage:
python -m eval.run_eval --config configs/b1_v1.yaml --split dev --limit 2

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

from src.config import load_config
from src.generation.answer import answer


def make_jsonable(value):
    """Convert dataclass results into JSON-compatible data."""

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

    with path.open("a", encoding="utf-8") as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=str
            ) + "\n"
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/b1_v1.yaml"
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