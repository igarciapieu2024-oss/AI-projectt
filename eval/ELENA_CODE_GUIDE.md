# Elena — Evaluation code walkthrough

These Python files contain documentation-only edits. Function logic, parameters,
imports, and evaluation outputs are unchanged.

## What to be able to explain

1. **`run_eval.py` — reproducibility and splits.** The CLI chooses a config and
   dev/test split, processes each question, passes the first turn as history for
   follow-up questions, and immediately persists results in JSONL. A held-out
   test run requires explicit `--allow-test`. The separate `evaluate()` helper
   exists for isolated unit tests and always blocks the test split.
2. **`metrics.py` — what is actually measured.** In the production report,
   question-level Recall@K is a binary hit if any top-k retrieved chunk covers
   an annotated gold ID; MRR discounts the rank of the first matching chunk.
   Citation *validity* checks label/chunk consistency, not whether claims are
   supported. Refusal metrics rely on the engine's boolean refusal flag.
3. **`judge.py` — factual correctness.** The production judge scores answers
   `0`, `0.5`, or `1`: deterministic rules handle refusals/unanswerable cases,
   while Ollama judges substantive answers against reference facts. A separate
   helper API uses **1–5**, only for claim-level compatibility/tests. Never
   combine those two scales in one result table.
4. **`error_analysis.py` — diagnostic, not ground truth.** Assigns a primary
   error type from judge results and retrieval/refusal/citation signals.
   Labels such as `generation_error` indicate where to inspect, not a proven
   root cause. Manual review corrected five dev judgments.
5. **`validate_questions.py` — dataset integrity.** Checks expected columns,
   question types, author quotas, duplicate questions, split assignment, and
   whether gold IDs actually exist in the parsed knowledge base.
6. **`figures.py` — reporting.** Reads previously saved summary files, then
   plots test results. The no-RAG B0 system is omitted from retrieval charts
   because retrieval metrics are not applicable to it.
7. **`find_ids.py` — annotation support.** Searches sections, slides, and image
   OCR descriptions for candidate source IDs; this literal search is not the
   production RAG retriever.
8. **`bootstrap.py` — uncertainty.** Resamples question scores to estimate
   confidence intervals; paired comparisons align identical question IDs.
9. **`benchmark_llm.py` — model speed check.** Times the same fixed task
   across available Ollama models; speed must still be considered alongside
   factual and citation quality.
10. **`run_all.py` — orchestration.** Thin command-line entry point to run
    multiple configurations; it is not a separate evaluation metric.

## How to put the files back

Replace the corresponding `.py` files under `AI-project/eval/` on your Mac.
Do not replace the `eval/runs` directory or question CSVs.

Then verify from the repository root:

```bash
python -m pytest -q
git diff --check
git diff --stat
```

Review the changes before committing. They should consist of documentation,
not new evaluation results or logic changes. Do not rerun the held-out test set.
