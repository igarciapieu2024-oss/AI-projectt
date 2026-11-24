# Grade whether each answer of an evaluation run is correct, against the reference answer of its question.
#   python -m eval.judge --run-dir eval/runs/RUN_ID [--rubric v1|v2|v3] [--allow-test]
# Refusals and answers to unanswerable questions are graded with fixed rules. Every other answer is graded by a local
# LLM (qwen2.5:7b, at temperature 0) with a score of 1 (correct), 0.5 (partly correct) or 0 (wrong). The search and
# the citations aren't graded here; eval/metrics.py measures them.
# It writes judge_questions.csv (one verdict per question) and judge_summary.json in the folder of the run; for the v2
# and v3 rubrics, the name of the rubric is added to both file names. eval/error_analysis.py and
# eval/judge_agreement.py read them.
# build_judge_messages() and judge_answer() are an older grading from 1 to 5, which we keep because the judge option
# of evaluate() in eval/run_eval.py and tests/test_judge.py use it.

from __future__ import annotations

# argparse reads the options, csv writes the verdicts, json reads the answers and the judge's reply, and re finds
# patterns in the text.
import argparse
import csv
import json
import re

# mean averages the scores.
from pathlib import Path
from statistics import mean


# Rubric v1 (the default): the answer is compared with the reference answer and the key facts of its question.
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

# Rubric v2: the answer is judged on what the QUESTION asks. The v1 rubric gave 0 to answers that were right but
# didn't repeat details of the reference that the question never asked for ("25% of the grade" without "only the four
# best quizzes count"). v1 stays the default, so earlier results can be reproduced; v2 is applied to every
# configuration in the same way and reported next to v1.
SYSTEM_V2 = (
    "You are an impartial evaluator of a university course Q&A system. "
    "Assess ONLY factual correctness against the supplied reference answer. "
    "Do not reward confident guesses. "
    "Ignore writing style and citation format; those are measured separately. "
    "The candidate may paraphrase correctly. "
    'Return JSON with keys "correct" (true or false), '
    '"score" (0, 0.5, or 1), and "reason" (one short sentence). '
    "Judge the answer on what the QUESTION asks. "
    "Score 1 if the candidate gets right everything the question asks and states nothing that "
    "contradicts the reference; details in the reference that the question does not ask for are not required, "
    "and extra correct information is not penalized. "
    "Score 0.5 if part of what the question asks is missing or wrong. "
    "Score 0 if the main fact asked for is wrong, missing, or contradicts the reference."
)


# Rubric v3: like v2, but the judge also reads the gold source (the slide or syllabus section the question was written
# from). Compared with the 80 dev answers graded against their sources (eval/reference_labels_dev.csv,
# eval/judge_agreement.py), v1 marked 9 correct answers as wrong and none the other way: it asked for reference
# details the question didn't ask for (EL01, FRA05, EL02), and it couldn't accept an answer that differed from the
# reference but matched the slide ("Session 7-8" in SA06, "RBI and Putouts" in EL13).
SYSTEM_V3 = (
    "You are an impartial evaluator of a university course Q&A system. "
    "You get the question, a reference answer, the gold source text the question was written from, "
    "and a candidate answer. Ignore writing style and citation format. "
    'Return JSON with keys "correct" (true or false), '
    '"score" (0, 0.5, or 1), and "reason" (one short sentence). '
    "Judge only what the QUESTION asks; details of the reference that the question does not ask for "
    "are not required. An answer that differs from the reference is correct if the gold source supports it. "
    "Extra information is not penalized unless it contradicts the gold source. "
    "Score 1 if everything the question asks is answered correctly and nothing contradicts the gold source. "
    "Score 0.5 if part of what the question asks is missing or wrong. "
    "Score 0 if the main thing asked for is wrong or missing, or the answer contradicts the gold source."
)

# The texts of all the sources, which we load the first time v3 needs them and then keep for the rest of the run.
_GOLD_TEXTS: dict[str, str] | None = None


# The text of every slide and syllabus section by id. We add the text read from the images of each slide
# (data/image_texts.jsonl) to its slide, so the answer to an image question is in its gold text too.
def gold_texts() -> dict[str, str]:
    # global lets the function fill the variable defined above, so the files are only read once.
    global _GOLD_TEXTS
    if _GOLD_TEXTS is None:
        # id -> text.
        texts: dict[str, str] = {}
        interim = Path("data/interim")
        # First the slides and the syllabus sections, from the parsed files.
        for name, key in (("slides.jsonl", "slide_id"), ("syllabus_sections.jsonl", "section_id")):
            path = interim / name
            if path.exists():
                for line in path.open(encoding="utf-8"):
                    item = json.loads(line)
                    texts[item[key]] = item.get("text", "")
        # Then the text of the images, added to the text of their slide.
        images = Path("data/image_texts.jsonl")
        if images.exists():
            for line in images.open(encoding="utf-8"):
                item = json.loads(line)
                extra = " ".join(x for x in (item.get("ocr_text"), item.get("description")) if x)
                if extra and item.get("slide_id") in texts:
                    texts[item["slide_id"]] += "\n[Image] " + extra
        _GOLD_TEXTS = texts
    return _GOLD_TEXTS


# The gold sources of one question as text for the v3 prompt, each cut to 4,000 characters (the longest gold
# section of the evaluation questions has 3,979).
def gold_source_text(gold_ids: str, per_source: int = 4000) -> str:
    # All the texts, loaded once.
    texts = gold_texts()
    # One block per gold id (separated by ";" or ","): the id in brackets and its text.
    parts = [
        f"[{gid.strip()}] {texts.get(gid.strip(), '(not found)')[:per_source]}"
        for gid in re.split(r"[;,]", gold_ids or "")
        if gid.strip()
    ]
    return "\n\n".join(parts)


# The prompt of the older grading from 1 to 5 (used by judge_answer below): it asks for a score and for the list of
# claims of the answer, each marked as supported by the sources or not. It isn't the 0 / 0.5 / 1 grading of main().
def build_judge_messages(
    answer_text: str,
    reference_answer: str,
    sources: list[dict],
) -> list[dict[str, str]]:

    # The instructions for the judge.
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

    # The sources the system used, each with its label, its ids and its text.
    evidence = []

    for source in sources:
        evidence.append(
            f"Label: {source.get('label', '')}\n"
            f"Source IDs: {source.get('covers', [])}\n"
            f"Text: {source.get('text', '')}"
        )

    # The reference answer, the answer to grade and the sources.
    user_content = (
        f"Reference answer:\n{reference_answer}\n\n"
        f"Candidate answer:\n{answer_text}\n\n"
        f"Retrieved evidence:\n"
        + "\n\n".join(evidence)
    )

    # The two messages: the instructions and the material to grade.
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


# Check that every label cited in the answer ([S1], [S2]...) is the label of one of the sources given. It doesn't
# check whether that source supports the sentence.
def check_citations(
    answer_text: str,
    sources: list[dict],
) -> dict:

    # The labels of the sources that were given, in capitals.
    valid_labels = {
        str(source.get("label", "")).upper()
        for source in sources
    }

    # The labels the answer cites, each one once and in order.
    cited_labels = list(dict.fromkeys(
        label.upper()
        for label in re.findall(
            r"\[(S\d+)\]",
            answer_text,
            flags=re.IGNORECASE,
        )
    ))

    # The cited labels that aren't among the sources.
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


# The older grading from 1 to 5: qwen2.5:7b gives the score and the claims, we check the reply, and we calculate the
# share of claims that aren't supported by the sources (hallucination_rate). It's separate from the 0 / 0.5 / 1 grades
# written by main(). chat_fn lets the tests pass a fake model.
def judge_answer(
    answer_text: str,
    reference_answer: str,
    sources: list[dict],
    cfg,
    chat_fn=None,
) -> dict:

    # replace() gives a copy of the config with one group changed.
    from dataclasses import replace

    # Without a chat function, the real one is used.
    if chat_fn is None:
        from src.generation.llm import chat
        chat_fn = chat

    # The same settings, with the judge model instead of the answer model.
    judge_cfg = replace(
        cfg,
        generation={
            **cfg.generation,
            "llm_model": "qwen2.5:7b",
        },
    )

    # Build the prompt and ask the judge for a JSON reply.
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

    # The reply has to be a JSON object with a whole score from 1 to 5 and a list of claims; anything else is an
    # error.
    try:
        response = json.loads(str(raw))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Judge returned invalid JSON") from exc

    if not isinstance(response, dict):
        raise ValueError("Judge returned invalid JSON object")

    # The score has to be a whole number from 1 to 5 (True and False are rejected too).
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

    # The claims have to be a list, and each claim needs its text and whether it's supported.
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

    # The claims marked as not supported, and their share of all the claims (None if there are no claims).
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

    # Everything the judge said, the counts, and the check of the cited labels.
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

# Read the judge's reply: a JSON object with a score of 0, 0.5 or 1 and a reason. A model sometimes wraps the JSON in
# a ``` block or writes text around it, so we remove the block marks and, if the text still isn't JSON, we read the
# part between the first { and the last }.
def parse_json_response(raw: str) -> dict:

    # The reply without spaces at the ends.
    text = raw.strip()

    # Remove the ``` marks of a code block, if there are any.
    if text.startswith("```"):

        text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            text,
            flags=re.I
        )

    # Try to read the text as JSON.
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

    # The reply has to be a JSON object.
    if not isinstance(obj, dict):
        raise ValueError(
            "Judge output must be a JSON object"
        )

    # The score has to be a number (True and False are rejected), and one of 0, 0.5 or 1.
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

    # And there has to be a reason, as text.
    if not isinstance(obj.get("reason"), str):
        raise ValueError(
            "Judge must provide a reason"
        )

    # The verdict: the score, whether it is fully correct, and the reason.
    return {
        "score": score,
        "correct": score == 1.0,
        "reason": obj["reason"]
    }


# Phrases the answer model uses to say that the sources don't contain what was asked (used by judge_one).
_NOT_IN_MATERIAL = re.compile(
    r"\b(not (explicitly |directly |specifically )?(covered|specified|mentioned|provided|included|stated|available|found)"
    r"|no (information|mention|data|details?)"
    r"|(does|do) not (provide|mention|specify|include|contain|state|cover)"
    r"|cannot be determined|isn't covered|is not in the (course )?material)\b",
    re.IGNORECASE,
)


# True when the answer says that the sources don't contain what was asked.
def says_not_in_material(answer: str) -> bool:
    return bool(_NOT_IN_MATERIAL.search(answer or ""))


# Grade one answer: a score of 1, 0.5 or 0, a reason, and the method used. Refusals, answers to unanswerable questions
# and empty answers follow fixed rules based on the answerable label of the question; only real answers to answerable
# questions are sent to the LLM.
def judge_one(row: dict, model: str, rubric: str = "v1") -> dict:

    # The answer of the system for this question.
    result = row.get("result") or {}

    # The facts we need: whether the question is answerable, whether the system refused and the text of the answer.
    answerable = (
        str(row.get("answerable", "")).strip() == "1"
    )

    refused = bool(
        result.get("refused")
    )

    answer = str(
        result.get("answer_text") or ""
    ).strip()

    # A refusal is right when the question is unanswerable and wrong when it's answerable. There is only one right
    # decision, so we use a fixed rule and don't spend an LLM call on it.

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

    # An unanswerable question answered by saying, in the model's own words, that the material doesn't contain it ("is
    # not specified in the provided sources") is a correct refusal. The refused flag of the answer phase only marks
    # the exact refusal sentence, so without this rule these answers would be graded as wrong. Answerable questions
    # are never graded this way.

    if not answerable and says_not_in_material(answer):

        return {
            "score": 1.0,
            "correct": True,
            "reason": "Semantic refusal: the answer says the material does not contain it",
            "judge_method": "semantic_refusal"
        }

    # Any other answer to an unanswerable question is wrong.

    if not answerable:

        return {
            "score": 0.0,
            "correct": False,
            "reason": "Answered a question marked unanswerable",
            "judge_method": "deterministic"
        }

    # An empty answer to an answerable question is wrong.

    if not answer:

        return {
            "score": 0.0,
            "correct": False,
            "reason": "Empty answer",
            "judge_method": "deterministic"
        }

    # The LLM judge, at temperature 0, so the same answer gets the same grade every time. The reference answer and the
    # key facts say what the answer has to contain; the search and the citation format aren't graded.

    # Imported here, so the fixed rules above work without Ollama.
    import ollama

    # The prompt for rubrics v1 and v2: the question, the reference answer, the key facts and the answer to grade.
    prompt = (
        f"Question:\n{row.get('question', '')}\n\n"
        f"Reference answer:\n{row.get('reference_answer', '')}\n\n"
        f"Essential key facts:\n{row.get('key_facts', '')}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        "Return only the requested JSON object."
    )
    if rubric == "v3":
        # v3 gives the gold source text instead of the key facts, which made the judge ask for every fact. For a
        # follow-up, we add the previous question, so the judge knows what the question refers to.
        prompt = (
            f"Question:\n{row.get('question', '')}\n\n"
            + (f"Previous question in the conversation:\n{row.get('turn1')}\n\n" if row.get("turn1") else "")
            + f"Reference answer:\n{row.get('reference_answer', '')}\n\n"
            f"Gold source text:\n{gold_source_text(str(row.get('gold_ids', '')))}\n\n"
            f"Candidate answer:\n{answer}\n\n"
            "Return only the requested JSON object."
        )
    # The instructions of the chosen rubric (v1 if the name is unknown).
    system = {"v2": SYSTEM_V2, "v3": SYSTEM_V3}.get(rubric, SYSTEM)

    # Ask the judge model for a JSON reply of at most 350 tokens.
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json",
        options={
            "temperature": 0,
            "num_predict": 350,
            # A larger context window for v3, with room for up to four gold sources; v1 and v2 keep the default.
            **({"num_ctx": 8192} if rubric == "v3" else {})
        }
    )

    # Depending on the version of the ollama library, the reply is a dictionary or an object.
    if isinstance(response, dict):

        raw = response["message"]["content"]

    else:

        raw = response.message.content

    # Read the verdict from the reply and mark it as graded by the LLM.
    verdict = parse_json_response(raw)

    verdict["judge_method"] = "llm"

    return verdict


# Grade every answer of a saved run and save the verdicts and the summary in the run folder.
def main():

    # The options: the folder of the run, the judge model, a limit, --allow-test and the rubric.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True
    )

    # The judge model, which is a different model from the one that answers.
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

    # The rubric: SYSTEM (v1), SYSTEM_V2 or SYSTEM_V3 above.
    parser.add_argument(
        "--rubric",
        choices=["v1", "v2", "v3"],
        default="v1"
    )

    args = parser.parse_args()

    # --limit has to be at least 1.
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    # Load the saved answers of the run (results.jsonl, written by eval/run_eval.py).

    run_path = (
        args.run_dir / "results.jsonl"
    )

    with run_path.open(encoding="utf-8") as file:

        rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    # Protect the test set, as in eval/run_eval.py: test answers are only graded with --allow-test.

    if not args.allow_test:

        if any(
            row.get("split") == "test"
            for row in rows
        ):

            parser.error(
                "Test set is protected"
            )

    # We only grade the questions that were answered without an error.

    rows = [
        row
        for row in rows
        if row.get("status") == "ok"
    ]

    # With --limit, only the first answers.
    if args.limit is not None:
        rows = rows[:args.limit]

    # Without any answer there is nothing to grade.
    if not rows:
        parser.error(
            "No successful answers in run"
        )

    # One verdict per answer.
    output = []

    # Grade every answer. If the judge fails on one, we save it with status "error" and the run goes on.

    for i, row in enumerate(rows, start=1):

        # The id of the question; print it so we can follow the run.
        qid = row.get("question_id", "")

        print(
            f"[{i}/{len(rows)}] Judging {qid}",
            flush=True
        )

        try:

            verdict = judge_one(
                row,
                args.model,
                args.rubric
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

    # The summary: the average score, the share of fully correct answers (score 1), and how many answers were graded
    # by a fixed rule and how many by the LLM.

    # The scores of the answers that were graded.
    scores = [
        row["score"]
        for row in output
        if row["status"] == "ok"
    ]

    summary = {
        "run_dir": str(args.run_dir),

        "judge_model": args.model,
        "rubric": args.rubric,

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

    # Save the verdict of each question (judge_questions.csv for v1, judge_questions_v2.csv or _v3.csv).

    csv_path = (
        args.run_dir / ("judge_questions.csv" if args.rubric == "v1" else f"judge_questions_{args.rubric}.csv")
    )

    # Write one row per verdict, with these columns.
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

    # Save the summary (judge_summary.json, with the same suffix for v2 and v3).

    json_path = (
        args.run_dir / ("judge_summary.json" if args.rubric == "v1" else f"judge_summary_{args.rubric}.json")
    )

    json_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        ) + "\n",
        encoding="utf-8"
    )

    # Print the summary.
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

    # Leave with an error code if any answer couldn't be graded.
    if len(output) != len(scores):
        raise SystemExit(1)


# Runs only when the file is executed directly.
if __name__ == "__main__":
    main()
