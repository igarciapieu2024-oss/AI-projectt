"""
Evaluate RAG answer correctness against reference facts.

The production CLI uses deterministic refusal checks and an Ollama judge
with scores 0 / 0.5 / 1. A separate 1-5 claim-level helper interface is
retained for compatibility with unit tests.

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


def build_judge_messages(
    answer_text: str,
    reference_answer: str,
    sources: list[dict],
) -> list[dict[str, str]]:
    """Build the auxiliary 1-5 claim-level evaluation prompt.

    This is not the 0 / 0.5 / 1 prompt used by the production run judge.
    """

    system = (
        "You are an impartial evaluator of a university course Q&A system. "
        "Compare the candidate answer against the reference answer and "
        "the supplied evidence. Do not invent facts. "
        "Return ONLY a JSON object with: "
        "'correctness' (integer 1 to 5), "
        "'claims' (list of objects with 'text' and 'supported'), "
        "and 'reasoning' (string). "
        "Use correctness 5 for a fully correct answer and 1 for an "
        "incorrect answer. Evaluate each factual claim independently."
    )

    evidence = []

    for source in sources:
        evidence.append(
            f"Label: {source.get('label', '')}\n"
            f"Source IDs: {source.get('covers', [])}\n"
            f"Text: {source.get('text', '')}"
        )

    user_content = (
        f"Reference answer:\n{reference_answer}\n\n"
        f"Candidate answer:\n{answer_text}\n\n"
        f"Retrieved evidence:\n"
        + "\n\n".join(evidence)
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def check_citations(
    answer_text: str,
    sources: list[dict],
) -> dict:
    """Check citation label existence, not evidential support of claims."""

    valid_labels = {
        str(source.get("label", "")).upper()
        for source in sources
    }

    cited_labels = list(dict.fromkeys(
        label.upper()
        for label in re.findall(
            r"\[(S\d+)\]",
            answer_text,
            flags=re.IGNORECASE,
        )
    ))

    invalid_labels = [
        label
        for label in cited_labels
        if label not in valid_labels
    ]

    return {
        "citation_labels_valid": len(invalid_labels) == 0,
        "cited_labels": cited_labels,
        "invalid_labels": invalid_labels,
    }


def judge_answer(
    answer_text: str,
    reference_answer: str,
    sources: list[dict],
    cfg,
    chat_fn=None,
) -> dict:
    """Evaluate factual claims using the auxiliary 1-5 judge interface.

    The source-level unsupported-claim rate is distinct from the main
    per-question correctness scores written by the CLI.
    """

    from dataclasses import replace

    if chat_fn is None:
        from src.generation.llm import chat
        chat_fn = chat

    judge_cfg = replace(
        cfg,
        generation={
            **cfg.generation,
            "llm_model": "qwen2.5:7b",
        },
    )

    messages = build_judge_messages(
        answer_text,
        reference_answer,
        sources,
    )

    raw = chat_fn(
        messages,
        judge_cfg,
        json=True,
    )

    try:
        response = json.loads(str(raw))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Judge returned invalid JSON") from exc

    if not isinstance(response, dict):
        raise ValueError("Judge returned invalid JSON object")

    correctness = response.get("correctness")

    if (
        isinstance(correctness, bool)
        or not isinstance(correctness, int)
        or correctness < 1
        or correctness > 5
    ):
        raise ValueError(
            "Correctness must be an integer between 1 and 5"
        )

    claims = response.get("claims")

    if not isinstance(claims, list):
        raise ValueError("Claims must be a list")

    for claim in claims:
        if (
            not isinstance(claim, dict)
            or not isinstance(claim.get("text"), str)
            or not isinstance(claim.get("supported"), bool)
        ):
            raise ValueError("Invalid claim format")

    unsupported = [
        claim["text"]
        for claim in claims
        if not claim["supported"]
    ]

    n_claims = len(claims)
    n_unsupported = len(unsupported)

    hallucination_rate = (
        n_unsupported / n_claims
        if n_claims
        else None
    )

    return {
        "correctness": correctness,
        "claims": claims,
        "reasoning": response.get("reasoning", ""),
        "n_claims": n_claims,
        "unsupported_claims": unsupported,
        "n_unsupported_claims": n_unsupported,
        "hallucination_rate": hallucination_rate,
        "citation_check": check_citations(
            answer_text,
            sources,
        ),
    }

def parse_json_response(raw: str) -> dict:
    """Extract and validate the production judge's 0 / 0.5 / 1 JSON."""

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
    """Score one production answer via deterministic checks or Ollama.

    Refusals and empty answers use explicit answerability labels, while
    substantive answers are compared with reference facts by the LLM.
    The production correctness scale is 0, 0.5, or 1.
    """

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

    # Refusal correctness follows the gold answerability label. Using a
    # deterministic rule here avoids spending judge-model calls on a case
    # with an unambiguous expected decision.

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

    # A non-refusal to an unanswerable question is penalized even if the
    # response text actually says the information is missing. This known
    # limitation of the engine's boolean flag motivated manual review.

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

    # Fix temperature to zero to reduce variation across judge runs. The
    # reference answer and key facts define the required factual content;
    # the judge intentionally does not evaluate retrieval or citation syntax.

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
    """Judge a saved run and write individual scores plus aggregate metrics."""

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
