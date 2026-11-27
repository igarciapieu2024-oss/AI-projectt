# Component-level evaluation of one saved run: the metrics of the RAG evaluation framework that eval/metrics.py and
# eval/judge.py don't calculate. Each metric looks at one part of the pipeline, so a low value says which part fails.
#   - Retrieval, context precision: of the sources the search gave to the LLM, how many help to answer the question.
#     eval/metrics.py already measures whether the right source is among them (recall@5 and MRR); this one measures
#     how much of what comes with it is noise.
#   - Generation, faithfulness: the share of the claims of the answer that the sources support. The citation check of
#     eval/metrics.py only checks that the cited label exists, not that the source says what the answer says.
#   - Generation, correctness with the right context: the share of correct answers among the questions whose gold
#     source was among the sources. It separates the mistakes of the LLM from the mistakes of the search.
#   - End to end, answer semantic similarity: the cosine similarity between the embedding of the answer and the
#     embedding of the reference answer.
# The first two are graded by the judge model (qwen2.5:7b, in Ollama); the other two don't call any LLM.
# It also calculates the ceilings that the design of the system puts on the retrieval metrics, so each result can be
# read against the best value it could have reached:
#   - the precision ceiling: the system always gives the LLM a fixed number of sources, and only a few chunks of the
#     knowledge base contain the answer to a question, so most of those places can't be filled with a right source;
#   - the recall ceiling: a question refused before searching, or whose gold source was left out of the indexes as a
#     near-duplicate, can't have its gold source among the sources.
#   python -m eval.component_metrics --run-dir eval/saved_runs/RUN_ID --allow-test [--skip-judge]
# With --skip-judge the judge isn't called: the verdicts saved by an earlier run are kept, and the metrics that need no
# LLM are calculated again (the judge takes minutes, the rest takes seconds).
# It only reads results.jsonl, the chunks of the knowledge base and the reference files, and it writes two files in
# the folder of the run (component_questions.csv and component_summary.json). No other file is changed.

from __future__ import annotations

# argparse reads the options, csv reads the grades and writes the metrics of each question, json reads the judge's
# reply and writes the averages, and re removes the citation labels from the answers.
import argparse
import csv
import json
import re

# Path holds the paths of the run folder and of the reference files.
from pathlib import Path

# load_scores reads the grades the automatic judge saved for a run (question_id -> grade).
from eval.judge_agreement import load_scores

# average gives the mean rounded to 4 decimals (None if there are no values), load_run reads results.jsonl,
# score_question gives recall@5 for one question and source_ids gives the slides or sections a retrieved chunk
# covers, so the averages and "the right source" mean here exactly what they mean in eval/metrics.py.
from eval.metrics import average, load_run, score_question, source_ids

# load_config gives the names of the judge model and of the embedding model, and chunks_path the file with the chunks
# of a knowledge base, so they are the same as in the rest of the project.
from src.config import chunks_path, load_config


# The instructions for the judge when it grades the search. A source counts as relevant only if it helps to answer
# the question: a source about the same topic that doesn't contain the answer is noise for the LLM.
RELEVANCE_SYSTEM = (
    "You are an impartial evaluator of the search step of a university course Q&A system. "
    "You get a question and the numbered sources that the search returned for it. "
    "Decide for each source whether it is relevant: it contains information that helps to answer the question. "
    "A source about the same topic that does not help to answer the question is not relevant. "
    'Return JSON with keys "relevant" (the list of the labels of the relevant sources, an empty list if none is '
    'relevant) and "reason" (one short sentence).'
)

# The instructions for the judge when it grades the answer against the sources. The judge doesn't get the reference
# answer and must not use what it knows: faithfulness is only about whether the sources support the answer, not about
# whether the answer is true (that is correctness, graded by eval/judge.py).
FAITHFULNESS_SYSTEM = (
    "You are an impartial evaluator of the answer step of a university course Q&A system. "
    "You get a question, the numbered sources the system retrieved, and the answer it wrote from them. "
    "Split the answer into its factual claims and decide for each claim whether the sources support it. "
    "A claim is supported only if the sources state it or it follows directly from them. "
    "Do not use your own knowledge and do not judge whether the claim is true. "
    "Ignore the citation labels such as [S1]. "
    'Return JSON with the key "claims": a list of objects with "text" (the claim) and "supported" (true or false).'
)

# The columns of component_questions.csv that the judge fills in, with the type of each one. With --skip-judge they
# are copied from the file saved by an earlier run.
JUDGE_COLUMNS = {
    "context_relevant": int,
    "context_precision": float,
    "relevant_labels": str,
    "claims": int,
    "supported_claims": int,
    "faithfulness": float,
    "unsupported_claims": str,
    "judge_status": str,
}


# Read the reference grades of one run: question_id -> 1 (correct), 0.5 (partly correct) or 0 (wrong). The file holds
# the grades of several runs, so we only keep the rows of this one. utf-8-sig also accepts files saved by Excel.
def load_grades(path: Path, run_id: str) -> dict[str, float]:
    # question_id -> grade.
    grades = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["run_id"] == run_id:
                grades[row["question_id"]] = float(row["reference_score"])
    return grades


# Read the reference check of the faithfulness of one run: question_id -> 1 if every claim of the answer is supported
# by its sources, 0 if at least one isn't. Each answer was compared with the sources the LLM received by a second
# model, the same way as the reference grades. It is used to check the judge; a missing file gives an empty result.
def load_faithfulness_check(path: Path, run_id: str) -> dict[str, int]:
    # question_id -> 1 or 0.
    checked = {}
    if not path.is_file():
        return checked
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["run_id"] == run_id:
                checked[row["question_id"]] = int(row["fully_supported"])
    return checked


# Read the judge columns saved by an earlier run: question_id -> the values of JUDGE_COLUMNS. An empty value stays
# empty, and the others get their type back (a CSV file stores everything as text).
def load_saved_judge(path: Path) -> dict[str, dict]:
    # question_id -> judge columns.
    saved = {}
    if not path.is_file():
        return saved
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            saved[row["question_id"]] = {
                name: (kind(row[name]) if row.get(name, "") != "" else "")
                for name, kind in JUDGE_COLUMNS.items()
            }
    return saved


# The slides or sections covered by each chunk that is in the indexes of a knowledge base. Near-duplicates are left
# out, because they are never indexed (src/kb/dedup.py), so the search can't return them. A missing file gives an
# empty list, and the ceilings are then left empty.
def indexed_covers(kb_sig: str) -> list[set[str]]:
    # One set of ids per indexed chunk.
    covers = []
    path = chunks_path(kb_sig)
    if not path.is_file():
        return covers
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunk = json.loads(line)
                if not chunk.get("duplicate_of"):
                    covers.append(set(chunk.get("covers") or []))
    return covers


# The gold ids of a question: the slides or sections that contain its answer, separated by ";".
def gold_ids(row: dict) -> set[str]:
    return {
        item.strip()
        for item in str(row.get("gold_ids") or "").split(";")
        if item.strip()
    }


# The question as the system searched it. For a follow-up we take the rewritten question, which makes sense without
# the previous turn; for the other questions it is the same as the original.
def question_text(row: dict) -> str:
    result = row["result"]
    return str(result.get("rewritten_question") or row.get("question") or "").strip()


# The sources of one answer as text for the judge, written like in the prompt of the answer (src/generation/prompts.py):
# the label, the position of the source in the course, and the text of the chunk.
def sources_block(hits: list[dict]) -> str:
    blocks = []
    for hit in hits:
        chunk = hit.get("chunk") or {}
        blocks.append(
            f"[{hit.get('label', '')}] {chunk.get('outline_path', '')}\n{str(chunk.get('text', '')).strip()}"
        )
    return "\n\n".join(blocks)


# The answer without its citation labels ([S1], [S2]...), which aren't part of what the answer says and would lower
# its similarity with the reference answer.
def strip_labels(answer: str) -> str:
    # Remove each label together with the comma or space before it, and then the double spaces that are left.
    text = re.sub(r"[\s,]*\[S\d+\]", "", answer or "", flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


# Send one grading request to the judge model and return its reply as a dictionary. Temperature 0 gives the same
# verdict every time, format "json" forces a JSON reply, and num_ctx leaves room for the 10 sources of a cross-course
# question (the same value eval/judge.py uses for its rubric v3).
def ask_judge(system: str, prompt: str, model: str) -> dict:
    # Imported here, so the metrics that need no LLM work without Ollama.
    import ollama

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        format="json",
        options={"temperature": 0, "num_predict": 1000, "num_ctx": 8192},
    )

    # Depending on the version of the ollama library, the reply is a dictionary or an object.
    if isinstance(response, dict):
        raw = response["message"]["content"]
    else:
        raw = response.message.content

    # The reply has to be a JSON object; a reply cut by the token limit isn't valid JSON and raises an error here.
    reply = json.loads(raw)
    if not isinstance(reply, dict):
        raise ValueError("Judge output must be a JSON object")
    return reply


# Retrieval: the share of the sources given to the LLM that are relevant to the question, according to the judge.
# ask lets the tests pass a fake judge.
def context_precision(row: dict, model: str, ask=ask_judge) -> dict:
    # The sources of this answer and their labels.
    hits = row["result"].get("context_hits") or []
    labels = [str(hit.get("label", "")).upper() for hit in hits]

    # The question and its sources.
    prompt = (
        f"Question:\n{question_text(row)}\n\n"
        f"Sources:\n{sources_block(hits)}\n\n"
        "Return only the requested JSON object."
    )
    reply = ask(RELEVANCE_SYSTEM, prompt, model)

    # The reply has to be a list of labels, and every label has to be one of the sources we sent.
    relevant = reply.get("relevant")
    if not isinstance(relevant, list):
        raise ValueError("Judge must return the list of relevant labels")
    relevant = sorted({str(label).upper() for label in relevant})
    if any(label not in labels for label in relevant):
        raise ValueError(f"Judge returned a label that is not a source: {relevant}")

    # The relevant sources over all the sources of the answer.
    return {
        "context_relevant": len(relevant),
        "context_precision": round(len(relevant) / len(labels), 4),
        "relevant_labels": ";".join(relevant),
    }


# Generation: the share of the claims of the answer that the sources support, according to the judge. The claims that
# aren't supported are saved as text, so they can be reviewed by hand.
def faithfulness(row: dict, model: str, ask=ask_judge) -> dict:
    # The sources and the answer the LLM wrote from them.
    result = row["result"]
    hits = result.get("context_hits") or []

    # The question, its sources and the answer to grade.
    prompt = (
        f"Question:\n{question_text(row)}\n\n"
        f"Sources:\n{sources_block(hits)}\n\n"
        f"Answer:\n{str(result.get('answer_text') or '').strip()}\n\n"
        "Return only the requested JSON object."
    )
    reply = ask(FAITHFULNESS_SYSTEM, prompt, model)

    # The reply has to be a list with at least one claim, and each claim needs its text and whether it's supported.
    claims = reply.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("Judge must return a non-empty list of claims")
    for claim in claims:
        if (
            not isinstance(claim, dict)
            or not isinstance(claim.get("text"), str)
            or not isinstance(claim.get("supported"), bool)
        ):
            raise ValueError("Invalid claim format")

    # The supported claims over all the claims of the answer.
    supported = sum(claim["supported"] for claim in claims)
    unsupported = [claim["text"] for claim in claims if not claim["supported"]]
    return {
        "claims": len(claims),
        "supported_claims": supported,
        "faithfulness": round(supported / len(claims), 4),
        "unsupported_claims": " | ".join(unsupported),
    }


# End to end: the cosine similarity between each answer and the reference answer of its question, with the embedding
# model of the knowledge base. embed() returns vectors of length 1, so the cosine similarity is their dot product.
# embed_fn lets the tests pass a fake embedding model.
def answer_similarities(rows: list[dict], model_name: str, embed_fn=None) -> dict[str, float]:
    # Without questions there is nothing to embed.
    if not rows:
        return {}

    # Imported here, so the file can be imported without loading the embedding library.
    if embed_fn is None:
        from src.retrieval.embed import embed
        embed_fn = embed

    # One vector per answer and one per reference answer, in the same order as the questions.
    answer_vectors = embed_fn(
        [strip_labels(row["result"].get("answer_text") or "") for row in rows],
        model_name,
    )
    reference_vectors = embed_fn(
        [str(row.get("reference_answer") or "") for row in rows],
        model_name,
    )

    # question_id -> similarity.
    return {
        row["question_id"]: round(float((answer * reference).sum()), 4)
        for row, answer, reference in zip(rows, answer_vectors, reference_vectors)
    }


# The share of fully correct answers (grade 1) in a group of questions, or None if the group is empty.
def correct_rate(lines: list[dict]):
    return average([float(line["correctness"] == 1) for line in lines])


# Calculate the metrics of every question of a saved run, average them by component and save both files.
def main():

    # The options: the folder of the run, the file with the reference grades, the file with the reference check of the
    # faithfulness, k (the sources that count for recall), the judge model, --skip-judge to calculate only the metrics
    # that need no LLM, and --allow-test.
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("eval/reference_labels_holdout3.csv"))
    parser.add_argument("--faithfulness-labels", type=Path, default=Path("eval/reference_faithfulness_holdout3.csv"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--model", default=None)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()

    # k has to be at least 1.
    if args.k < 1:
        parser.error("--k must be at least 1")

    # Load the saved answers. We only use the questions that were answered without an error.
    rows = [
        row
        for row in load_run(args.run_dir / "results.jsonl")
        if row.get("status") == "ok"
    ]
    if not rows:
        parser.error("No successful answers in run")

    # Protect the test set, as in eval/judge.py: test answers are only measured with --allow-test. These metrics are
    # calculated from the saved answers and nothing is tuned with them, so the test set stays unseen.
    if not args.allow_test and any(row.get("split") == "test" for row in rows):
        parser.error("Test set is protected")

    # The reference grades of this run (the run id is the name of its folder), and the reference check of the
    # faithfulness, if this run has one. A new run has no reference grades, so its grades are the ones the automatic
    # judge saved in its folder (eval/judge.py), if it was run.
    grades = load_grades(args.labels, args.run_dir.name)
    if not grades and (args.run_dir / "judge_questions.csv").is_file():
        grades = load_scores(args.run_dir / "judge_questions.csv")
    if not grades:
        parser.error(f"No grades for run {args.run_dir.name}: run eval/judge.py on it first")
    checked = load_faithfulness_check(args.faithfulness_labels, args.run_dir.name)

    # The judge model and the embedding model, from the shared config.
    cfg = load_config("configs/base.yaml")
    judge_model = args.model or cfg.eval["judge_model"]
    embed_model = cfg.kb["embed_model"]

    # The two files this script writes. With --skip-judge, the verdicts of an earlier run are read from the first one
    # before it is written again.
    csv_path = args.run_dir / "component_questions.csv"
    json_path = args.run_dir / "component_summary.json"
    saved_judge = load_saved_judge(csv_path) if args.skip_judge else {}

    # The chunks that are in the indexes of the knowledge base of this run, to know how many of them contain the
    # answer to each question. The code of the knowledge base is saved with every answer.
    kb_sig = next((row["result"].get("kb_sig") for row in rows if row["result"].get("kb_sig")), None)
    index = indexed_covers(kb_sig) if kb_sig else []

    # The similarity with the reference answer, only for the answerable questions: the unanswerable ones have no
    # reference answer to compare with. A refused answerable question is kept, because for the student it is a
    # failure of the whole system, and its refusal sentence gets a low similarity.
    similarities = answer_similarities(
        [row for row in rows if str(row.get("answerable", "")).strip() == "1"],
        embed_model,
    )

    # One row of metrics per question.
    output = []

    for i, row in enumerate(rows, start=1):

        qid = row.get("question_id", "")

        # recall@5, the answerable label and the refusal flag, calculated as in eval/metrics.py.
        scores = score_question(row, args.k)
        hits = row["result"].get("context_hits") or []
        answerable = scores["answerable"] == 1
        refused = scores["refused"] == 1

        # Context precision needs an answerable question (an unanswerable one has no relevant source) that reached the
        # search. Faithfulness needs an answer the LLM wrote from sources, so refusals are left out.
        needs_precision = answerable and bool(hits)
        needs_faithfulness = bool(hits) and not refused

        # The row of this question. The judge metrics start empty and are filled in below.
        line = {
            "question_id": qid,
            "type": row.get("type", ""),
            "answerable": scores["answerable"],
            "refused": scores["refused"],
            "recall_at_k": scores["recall_at_k"],
            "correctness": grades.get(qid, ""),
            "semantic_similarity": similarities.get(qid, ""),
            "context_total": len(hits) if needs_precision else "",
            "gold_in_index": "",
            "gold_sources": "",
            "gold_precision": "",
            "precision_ceiling": "",
            "context_relevant": "",
            "context_precision": "",
            "relevant_labels": "",
            "claims": "",
            "supported_claims": "",
            "faithfulness": "",
            "unsupported_claims": "",
            "reference_fully_supported": checked.get(qid, ""),
            "judge_status": "skipped",
        }

        # The ceilings, from the gold ids of the question, without any model.
        gold = gold_ids(row)

        # How many chunks of the indexes cover a gold id: with 0, the search can never find the right source.
        if answerable and index:
            line["gold_in_index"] = sum(bool(gold & covers) for covers in index)

        if needs_precision:
            # The sources given to the LLM that cover a gold id, over all the sources: the precision by the gold ids.
            line["gold_sources"] = sum(bool(gold & source_ids(hit)) for hit in hits)
            line["gold_precision"] = round(line["gold_sources"] / len(hits), 4)

            # The best this value could be: a perfect search would return every chunk that covers a gold id, but it
            # still has to fill all the places, and there are fewer of those chunks than places.
            if index:
                line["precision_ceiling"] = round(min(len(hits), line["gold_in_index"]) / len(hits), 4)

        # With --skip-judge, keep the verdict an earlier run saved for this question.
        if args.skip_judge:
            line.update(saved_judge.get(qid, {}))

        elif not (needs_precision or needs_faithfulness):
            line["judge_status"] = "not_needed"

        else:
            print(f"[{i}/{len(rows)}] Judging {qid}", flush=True)

            # If the judge fails on one question, we save the error and the run goes on.
            try:
                if needs_precision:
                    line.update(context_precision(row, judge_model))
                if needs_faithfulness:
                    line.update(faithfulness(row, judge_model))
                line["judge_status"] = "ok"

            except Exception as exc:
                print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
                line["judge_status"] = f"error: {type(exc).__name__}: {exc}"

        output.append(line)

    # The groups of questions for the averages. with_context are the answerable questions whose gold source was among
    # the sources given to the LLM, so a wrong answer there is a mistake of the generation; without_context are the
    # ones where it wasn't (the search missed it, or the question was refused before searching).
    answerable_lines = [line for line in output if line["answerable"] == 1]
    with_context = [line for line in answerable_lines if line["recall_at_k"] == 1]
    without_context = [line for line in answerable_lines if line["recall_at_k"] == 0]
    correct_lines = [line for line in answerable_lines if line["correctness"] == 1]
    not_correct_lines = [line for line in answerable_lines if line["correctness"] != 1]

    # The answerable questions that reached the search, and the ones whose gold source can be found at all: they were
    # searched, and at least one chunk of the indexes covers a gold id.
    searched = [line for line in answerable_lines if line["context_total"] != ""]
    reachable = [line for line in searched if line["gold_in_index"] not in ("", 0)]

    # The questions the judge graded for each of its two metrics, and the answers that also have a reference check.
    precision_lines = [line for line in output if line["context_precision"] != ""]
    faithfulness_lines = [line for line in output if line["faithfulness"] != ""]
    checked_lines = [line for line in output if line["reference_fully_supported"] != ""]
    both_lines = [line for line in checked_lines if line["faithfulness"] != ""]

    # Where the judge results of this run come from.
    if not args.skip_judge:
        judge_source = "new"
    elif saved_judge:
        judge_source = "saved by an earlier run"
    else:
        judge_source = "none"

    # The averages, grouped by the component they evaluate. Each ceiling is next to the result it limits.
    summary = {
        "run_dir": str(args.run_dir),
        "judge_model": judge_model if judge_source != "none" else None,
        "judge_results": judge_source,
        "embed_model": embed_model,
        "evaluated": len(output),
        "judge_errors": sum(str(line["judge_status"]).startswith("error") for line in output),

        "retrieval": {
            "recall_at_k": average([line["recall_at_k"] for line in answerable_lines]),
            "recall_ceiling": (
                round(len(reachable) / len(answerable_lines), 4)
                if index and answerable_lines
                else None
            ),
            "answerable_not_searched": len(answerable_lines) - len(searched),
            "gold_source_not_in_index": (len(searched) - len(reachable)) if index else None,
            "context_precision": average([line["context_precision"] for line in precision_lines]),
            "gold_precision": average([line["gold_precision"] for line in searched]),
            "precision_ceiling": average(
                [line["precision_ceiling"] for line in searched if line["precision_ceiling"] != ""]
            ),
            "questions": len(searched),
        },

        "generation": {
            "faithfulness": average([line["faithfulness"] for line in faithfulness_lines]),
            "fully_supported_rate": average([float(line["faithfulness"] == 1) for line in faithfulness_lines]),
            "answers": len(faithfulness_lines),
            "reference_fully_supported_rate": average(
                [float(line["reference_fully_supported"]) for line in checked_lines]
            ),
            "judge_agreement_with_reference": average(
                [
                    float((line["faithfulness"] == 1) == (line["reference_fully_supported"] == 1))
                    for line in both_lines
                ]
            ),
            "correct_rate_with_gold_source": correct_rate(with_context),
            "questions_with_gold_source": len(with_context),
            "correct_rate_without_gold_source": correct_rate(without_context),
            "questions_without_gold_source": len(without_context),
        },

        "end_to_end": {
            "answerable_correct_rate": correct_rate(answerable_lines),
            "answer_semantic_similarity": average([line["semantic_similarity"] for line in answerable_lines]),
            "questions": len(answerable_lines),
            "similarity_of_correct_answers": average([line["semantic_similarity"] for line in correct_lines]),
            "similarity_of_other_answers": average([line["semantic_similarity"] for line in not_correct_lines]),
        },

        "notes": [
            "Context precision: relevant sources / sources given to the LLM, graded by the judge on answerable questions.",
            "Gold precision: sources that cover a gold id / sources given to the LLM. Precision ceiling: its best possible value with the same number of sources.",
            "Recall ceiling: answerable questions that were searched and whose gold source is in the indexes.",
            "Faithfulness: supported claims / claims of the answer, graded by the judge on the answers that weren't refused.",
            "Reference fully supported rate: answers whose claims are all supported, checked against the sources by a second model.",
            "Correct rate with gold source: fully correct answers among the answerable questions with recall@k = 1.",
            "Answerable correct rate = recall@k x correct rate with gold source, when no answer is correct without it.",
            "Answer semantic similarity: cosine between the answer and the reference answer, on answerable questions. The similarity of the correct answers is its practical ceiling.",
        ],
    }

    # Save the metrics of each question. The columns are the keys of the first row.
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output[0].keys()))
        writer.writeheader()
        writer.writerows(output)

    # Save the averages.
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Print the averages.
    print("\n========== COMPONENT METRICS ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=======================================")
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")

    # Leave with an error code if the judge couldn't grade some question.
    if summary["judge_errors"]:
        raise SystemExit(1)


# Runs only when the file is executed directly.
if __name__ == "__main__":
    main()
