"""
Evaluate RAG answer correctness using an LLM judge.

Owner: Elena.
"""

from __future__ import annotations

import argparse
import csv
import json
import re

from pathlib import Path
from statistics import mean


SYSTEM = (
    "You are an impartial evaluator of a university course Q&A system. "
    "Assess ONLY factual correctness against the supplied reference answer. "
    "Do not reward confident guesses. "
    "Ignore writing style and citation format; those are measured separately. "
    "The candidate may paraphrase correctly. "
    'Return JSON with keys "correct" (true or false), '
    '"score" (0, 0.5, or 1), and "reason" (one short sentence). '
    "Score 1 if all essential facts are correct; "
    "0.5 if partly correct without a major contradiction; "
    "0 if wrong, misleading, or unsupported."
)


def parse_json_response(raw: str) -> dict:

    text = raw.strip()

    if text.startswith("```"):

        text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            text,
            flags=re.I
        )

    try:

        obj = json.loads(text)

    except json.JSONDecodeError:

        match = re.search(
            r"\{.*\}",
            text,
            flags=re.S
        )

        if not match:
            raise ValueError(
                f"Judge did not return JSON: {raw[:200]}"
            )

        obj = json.loads(match.group())

    if not isinstance(obj, dict):
        raise ValueError(
            "Judge output must be a JSON object"
        )

    score = obj.get("score")

    if isinstance(score, bool) or not isinstance(
        score,
        (int, float)
    ):
        raise ValueError(
            "Judge score must be numeric"
        )

    score = float(score)

    if score not in {0.0, 0.5, 1.0}:
        raise ValueError(
            "Judge score must be 0, 0.5, or 1"
        )

    if not isinstance(obj.get("reason"), str):
        raise ValueError(
            "Judge must provide a reason"
        )

    return {
        "score": score,
        "correct": score == 1.0,
        "reason": obj["reason"]
    }


def judge_one(row: dict, model: str) -> dict:

    result = row.get("result") or {}

    answerable = (
        str(row.get("answerable", "")).strip() == "1"
    )

    refused = bool(
        result.get("refused")
    )

    answer = str(
        result.get("answer_text") or ""
    ).strip()

    # Evaluate refusals automatically

    if refused:

        score = 0.0 if answerable else 1.0

        return {
            "score": score,
            "correct": score == 1.0,
            "reason": (
                "False refusal on an answerable question"
                if answerable
                else "Correct refusal on an unanswerable question"
            ),
            "judge_method": "deterministic"
        }

    # Incorrect acceptance of unanswerable questions

    if not answerable:

        return {
            "score": 0.0,
            "correct": False,
            "reason": "Answered a question marked unanswerable",
            "judge_method": "deterministic"
        }

    # Empty response

    if not answer:

        return {
            "score": 0.0,
            "correct": False,
            "reason": "Empty answer",
            "judge_method": "deterministic"
        }

    # Evaluate substantive answers using Ollama

    import ollama

    prompt = (
        f"Question:\n{row.get('question', '')}\n\n"
        f"Reference answer:\n{row.get('reference_answer', '')}\n\n"
        f"Essential key facts:\n{row.get('key_facts', '')}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        "Return only the requested JSON object."
    )

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json",
        options={
            "temperature": 0,
            "num_predict": 350
        }
    )

    if isinstance(response, dict):

        raw = response["message"]["content"]

    else:

        raw = response.message.content

    verdict = parse_json_response(raw)

    verdict["judge_method"] = "llm"

    return verdict


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--model",
        default="qwen2.5:7b"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    parser.add_argument(
        "--allow-test",
        action="store_true"
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    # Load saved RAG results

    run_path = (
        args.run_dir / "results.jsonl"
    )

    with run_path.open(encoding="utf-8") as file:

        rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    # Protect the test set

    if not args.allow_test:

        if any(
            row.get("split") == "test"
            for row in rows
        ):

            parser.error(
                "Test set is protected"
            )

    # Keep successful executions

    rows = [
        row
        for row in rows
        if row.get("status") == "ok"
    ]

    if args.limit is not None:
        rows = rows[:args.limit]

    if not rows:
        parser.error(
            "No successful answers in run"
        )

    output = []

    # Judge every answer

    for i, row in enumerate(rows, start=1):

        qid = row.get("question_id", "")

        print(
            f"[{i}/{len(rows)}] Judging {qid}",
            flush=True
        )

        try:

            verdict = judge_one(
                row,
                args.model
            )

            output.append({
                "question_id": qid,
                "status": "ok",
                **verdict
            })

        except Exception as exc:

            print(
                f"ERROR: {type(exc).__name__}: {exc}",
                flush=True
            )

            output.append({
                "question_id": qid,
                "status": "error",
                "score": "",
                "correct": "",
                "reason": (
                    f"{type(exc).__name__}: {exc}"
                ),
                "judge_method": "failed"
            })

    # Aggregate results

    scores = [
        row["score"]
        for row in output
        if row["status"] == "ok"
    ]

    summary = {
        "run_dir": str(args.run_dir),

        "judge_model": args.model,

        "evaluated": len(scores),

        "failed": len(output) - len(scores),

        "mean_correctness_score": (
            round(mean(scores), 4)
            if scores
            else None
        ),

        "exact_correct_rate": (
            round(
                mean(score == 1 for score in scores),
                4
            )
            if scores
            else None
        ),

        "methods": {
            "deterministic": sum(
                row["judge_method"] == "deterministic"
                for row in output
            ),

            "llm": sum(
                row["judge_method"] == "llm"
                for row in output
            )
        },

        "note": (
            "Evaluates factual correctness. "
            "Retrieval and citations are scored separately."
        )
    }

    # Save individual judgments

    csv_path = (
        args.run_dir / "judge_questions.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question_id",
                "status",
                "score",
                "correct",
                "reason",
                "judge_method"
            ]
        )

        writer.writeheader()

        writer.writerows(output)

    # Save summary

    json_path = (
        args.run_dir / "judge_summary.json"
    )

    json_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        ) + "\n",
        encoding="utf-8"
    )

    print(
        "\n========== JUDGE RESULTS =========="
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        )
    )

    print(
        "==================================="
    )

    if len(output) != len(scores):
        raise SystemExit(1)


if __name__ == "__main__":
    main()