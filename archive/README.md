# AI-project — offline course assistant (RAG)

Answers students' questions about their IE courses using only the official syllabi and session slides (images included), fully offline: local embeddings, ChromaDB, a keyword index, a trained router, a reranker and a local LLM through Ollama, in a Streamlit chat with cited sources.

## Setup (once, with internet)

1. Install **Python 3.11+**, **Git**, **VS Code** and **[Ollama](https://ollama.com/download)**.
2. Clone the repo and create the environment:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Download the local models (names are in `configs/base.yaml`):

```bash
ollama pull llama3.2:3b      # answers, translation and follow-up rewriting (needed for the app)
ollama pull llama3.1:8b      # answers of the final 8B system (final_tables)
ollama pull qwen2.5:7b       # evaluation judge (only to run eval/)
ollama pull qwen2.5vl:7b     # image classification and descriptions (only to regenerate data/image_texts.jsonl)
```

The embedding model and the reranker are downloaded from Hugging Face the first time the knowledge base is built.

4. Check everything:

```bash
python -m src.check_env
```

## Run the evaluated version

The version evaluated on the 60 held-out test questions is the tag `v1.2-eval`. `main` contains the same system plus the evaluation code.

```bash
python -m src.pipeline build --config configs/final.yaml    # ~5 min, ends with kb_sig=ec71346e
python -m pytest -q                                          # all tests pass (2 skipped without a built index)
streamlit run app.py                                         # the app; the first answer takes ~30 s while models load
```

Results: `reports/evaluation_report.md` (test set: B0 11.7%, B1 26.7%, final 55.0% fully correct answers).

## Final system (18 Nov): final_tables

After v1.2, 11 configurations were compared on dev and the best was measured once on three new held-out sets of 30 questions (`eval/questions_holdout*.csv`). The final system, `configs/final_tables.yaml`, answers with `llama3.1:8b`, answers each named course separately, checks follow-up rewrites, reads slide tables row by row and uses a fixed seed.

```bash
python -m src.pipeline build --config configs/final_tables.yaml   # ends with kb_sig=586a05db
$env:APP_CONFIG="final_tables"; streamlit run app.py               # PowerShell; ~30 s an answer on a laptop CPU
```

| Unseen questions | Fully correct |
| --- | --- |
| final_tables, holdout 3 | 73.3% |
| 8B configuration (final_best2), holdouts 1–3 | 70.0% (63 of 90) |
| v1.2 (3B), holdout 3 | 56.7% |
| LLM alone, holdout 3 | 20.0% |

Grades checked against the gold sources are in `eval/reference_labels_*.csv`; `python -m eval.judge_agreement` compares them with the automatic judge. Only one Ollama model should be loaded at a time on a 16 GB laptop (`ollama stop <model>` between runs). Without `APP_CONFIG` the app runs `final` (3B, ~14 s an answer).

## Commands (run from the repo root)

| Job | Command | Owner |
| --- | --- | --- |
| Image text, once | `python -m src.vision.run_vision` | Santiago + Irene |
| Build the knowledge base | `python -m src.pipeline build --config configs/final.yaml` | Santiago |
| Train the router | `python -m src.train.train_router` | Francesca |
| Evaluate one config | `python -m eval.run_eval --config configs/final.yaml --split dev` | Elena |
| Compare local LLM speed | `python -m eval.benchmark_llm --runs 3` | Sergio |
| Start the app (fake engine) | `USE_FAKE=1 streamlit run app.py` (PowerShell: `$env:USE_FAKE=1; streamlit run app.py`) | Irene |
| Start the app (real engine) | `streamlit run app.py` | Irene |
| Retrieval benchmark (44 questions) | `python bench_retrieval.py --config configs/final.yaml` | Irene |
| Full answers for 8 fixed questions | `python diag_cite.py` | Irene |
| Fast tests | `pytest -m "not slow"` | everyone |

The LLM benchmark tests `llama3.1:8b`, `qwen2.5:7b` and `llama3.2:3b` with the same cited-answer prompt. Pull the models first with `ollama pull <model>`. It prints each response and elapsed time; judge answer quality and citation behavior as well as whether every run meets the 10-second target. The first run may include model loading time.

## Where things go

| Folder | What | Owner |
| --- | --- | --- |
| `configs/` | `base.yaml` + one file per system variant (`extends: base.yaml`) | shared; experiment files Elena |
| `data/raw/syllabi/`, `data/raw/slides/`, `data/manifest.csv` | the downloaded PDFs, one manifest row per PDF | Santiago |
| `data/labels/` | image labels · router questions | Irene · Francesca |
| `data/image_texts.jsonl` | text and description of every image | Santiago + Irene |
| `src/schemas.py`, `src/config.py`, `src/utils/`, `src/check_env.py`, `src/pipeline.py` | shared formats, settings, commands | Francesca |
| `src/kb/` | PDFs → clean → chunks → dedup → indexes, course table | Santiago |
| `src/vision/` | extract + OCR (Santiago) · classify + describe (Irene) | Santiago, Irene |
| `src/retrieval/`, `src/generation/` | search, reranker, router wrapper, prompts, LLM, `answer()` | Sergio |
| `src/train/` | image classifier · router | Irene · Francesca |
| `app.py`, `app_pages/`, `app_components/`, `app_content.py`, `.streamlit/`, `assets/`, `static/` | the Streamlit app (texts in `app_content.py`, look in `assets/style.css`, fonts in `static/fonts/`) | Irene |
| `eval/` | questions, harness, judge, metrics, statistics, figures | Elena |
| `reports/` | small result files used in the documents and slides | each owner |
| `docs/` | `parts/<name>.md`, technical and end-to-end documents | Francesca (parts from everyone) |

Generated folders (`chroma_db/`, `data/interim/`, `data/processed/`, `eval/runs/`) are built on each laptop and never uploaded.

## Rules

- Never work on `main`: one branch per task (`kb/…`, `rag/…`, `vision/…`, `app/…`, `eval/…`, `router/…`, `docs/…`), then a pull request with one approval and green tests.
- Shared formats live in `src/schemas.py`. To change one, open a pull request titled `CONTRACT: …` approved by everyone who reads that file.
- Every path comes from `PATHS` in `src/config.py`; every model name comes from the configs.
- Heavy libraries are imported inside functions, so `pytest -m "not slow"` runs without models.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `check_env` says Ollama is not running | Open the Ollama app, then run it again |
| A model is missing | `ollama pull <name>` (names in `configs/base.yaml`) |
| `answer()` says the index doesn't exist | `python -m src.pipeline build --config <the same config>` |
| Answers are slow | The first answer loads the models (~30 s); later ones take 10–16 s on a laptop CPU |
