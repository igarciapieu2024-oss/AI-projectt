# Connects the app to the answer engine and to the course list.
# Every page imports answer_stream and courses from here, so the switch between the fake and the real engine
# happens in one place only.
from __future__ import annotations  # lets us write types like "dict[str, str]" in any Python version

import csv  # reads the manifest
import os  # reads the USE_FAKE environment variable
from pathlib import Path  # handles file paths the same way on Windows and Mac

import streamlit as st  # for st.cache_data

import app_content as content  # the placeholder courses
from src.config import PATHS, load_config  # the shared folder paths and the settings loader

# USE_FAKE=1 -> the fake answer (src/generation/fake.py); otherwise the real engine (src/generation/answer.py).
# both have exactly the same inputs and outputs, so the pages don't need to know which one is running
USING_FAKE = os.environ.get("USE_FAKE") == "1"
# APP_CONFIG picks the system the assistant runs: final_system (our final 8B system, ~30 s an answer
# on a laptop CPU) by default; APP_CONFIG=baselines/v1_2_small_model runs the earlier 3B system (~14 s)
MAIN_CONFIG = os.environ.get("APP_CONFIG", "final_system")
if USING_FAKE:
    from src.generation.fake import answer_stream  # noqa: F401  (pages import it from here)
else:
    from src.generation.answer import answer_stream  # noqa: F401


# loads configs/<name>.yaml, e.g. "final_system" or "baselines/b0_llm_only"
@st.cache_resource  # loads each config once and keeps it, instead of reading the file on every click
def config(name: str):
    return load_config(f"configs/{name}.yaml")


# loads the models and the indexes of this configuration once, when the app starts (called by app.py);
# returns kb_sig, the name of the knowledge base that was loaded
@st.cache_resource(show_spinner="Loading local retrieval models...")
def warm_up_models(name: str = MAIN_CONFIG) -> str:
    from src.warmup import warm_up

    cfg = config(name)
    warm_up(cfg)
    return cfg.kb_sig


# course_code -> course name, from data/manifest.csv; the placeholders until it has rows
@st.cache_data  # reads the manifest once, not on every click
def courses() -> dict[str, str]:
    found: dict[str, str] = {}
    if PATHS["manifest"].exists():  # the manifest is written when the PDFs are added
        with PATHS["manifest"].open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):  # one row per PDF, so the same course appears several times
                code = row["course_code"]
                found[code] = content.COURSE_DISPLAY_NAMES.get(code, row["course_name"])  # the dict keeps each course once
    found = dict(sorted(found.items(), key=lambda item: item[1]))  # alphabetical, as in the filter menu
    return found or content.PLACEHOLDER_COURSES  # placeholders if the manifest is still empty


# every PDF under data/raw/, keyed by its path in lower case -> its real path.
# the manifest and the folders don't always use the same capitals ("presentations" vs "Presentations"); Windows
# and Mac don't mind, Linux does, so looking files up in lower case makes the app work on all three
@st.cache_data  # the folder is scanned once per app start
def _all_pdfs() -> dict[str, str]:
    raw = PATHS["root"] / "data" / "raw"
    return {p.relative_to(raw).as_posix().lower(): str(p) for p in raw.rglob("*.pdf")}


# the real path of a PDF named in the manifest, or None if it isn't on this laptop
def pdf_path(manifest_file: str) -> Path | None:
    found = _all_pdfs().get(manifest_file.replace("\\", "/").lower())
    return Path(found) if found else None


# what the assistant knows, counted from the real files: shown on the Home and How it works pages.
# None means "not available on this laptop yet" (e.g. the images before the image extraction runs)
@st.cache_data(show_spinner=False)  # counted once per app start (opening 86 PDFs takes a second)
def knowledge_base_stats() -> dict[str, int | None]:
    stats: dict[str, int | None] = {"courses": 0, "syllabi": 0, "decks": 0, "slides": None, "images": None}
    if not PATHS["manifest"].exists():
        return stats
    with PATHS["manifest"].open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))  # one row per PDF
    stats["courses"] = len({r["course_code"] for r in rows})
    stats["syllabi"] = sum(r["doc_type"] == "syllabus" for r in rows)
    stats["decks"] = sum(r["doc_type"] == "slides" for r in rows)

    try:  # slide pages: counted from the PDFs themselves (PyMuPDF only opens them, it doesn't read the text)
        import pymupdf  # heavy library: imported here, only for this count
        pages = 0
        for r in rows:
            path = pdf_path(r["file"]) if r["doc_type"] == "slides" else None
            if path is not None:
                with pymupdf.open(path) as pdf:
                    pages += pdf.page_count
        stats["slides"] = pages or None
    except Exception:  # PyMuPDF missing or a broken PDF: the page just shows "—" for this number
        stats["slides"] = None

    if PATHS["images_index"].exists():  # written by the image extraction, once
        with PATHS["images_index"].open(encoding="utf-8", newline="") as f:
            stats["images"] = sum(1 for _ in csv.DictReader(f))
    return stats
