"""Fake answer() with the same inputs and outputs as the real one (owner: Sergio).

The app and the evaluation use it until the real engine exists:
    USE_FAKE=1 streamlit run app.py
It never searches or calls an LLM. It returns made-up answers that cover every case the app must show:
- a normal answer with 5 sources (a syllabus section, slides, and a slide whose source is an image);
- a refusal, when the question looks out of scope (it mentions "party", "weather", "football"...);
- the bare-LLM answer with no sources, when the config has generation.rag = false (b0_no_rag.yaml).
"""
from __future__ import annotations

import time
from typing import Iterator

from src.schemas import REFUSAL, AnswerResult, Chunk, Citation, Filters, Hit, RouteDecision

# words that make the fake answer refuse, so the app can show what a refusal looks like
_OUT_OF_SCOPE_WORDS = ("party", "weather", "football", "capital of", "recipe")


def _chunk(chunk_id: str, doc_type: str, source_id: str, covers: list[str], sessions: list[int],
           outline: str, text: str, has_image_text: bool = False) -> Chunk:
    """Builds one fake source with every field the real ones have."""
    return Chunk(chunk_id=chunk_id, kb_sig="fake0000", strategy="slide" if doc_type == "slide" else "section",
                 doc_type=doc_type, course_code="c01", sessions=sessions, source_id=source_id, covers=covers,
                 parent_id=f"{source_id}_parent", outline_path=outline, text=text, embed_text=f"{outline}\n{text}",
                 has_image_text=has_image_text)


# 5 example sources: one syllabus section, three slides, and one slide answered from its image
_SOURCES = [
    _chunk("section__c01_syl_sec04__00", "syllabus_section", "c01_syl", ["c01_syl_sec04"], [],
           "Course 1 · Syllabus › EVALUATION",
           "Final exam: 40%. Group project: 30%. Class participation: 20%. Quizzes: 10%."),
    _chunk("slide__c01_s04_p017__00", "slide", "c01_s04", ["c01_s04_p017"], [4],
           "Course 1 · Session 4 › 2. Topic › 2.1 Subtopic",
           "Example slide text about the main concept of the session."),
    _chunk("slide__c01_s04_p018__00", "slide", "c01_s04", ["c01_s04_p018"], [4],
           "Course 1 · Session 4 › 2. Topic › 2.2 Example",
           "Slide text. [Image: diagram] OCR: \"Input → Step 1 → Step 2 → Output\" "
           "Description: A flow diagram with four boxes from left to right connected by arrows.",
           has_image_text=True),
    _chunk("slide__c01_s05_p003__00", "slide", "c01_s05", ["c01_s05_p003"], [5],
           "Course 1 · Session 5 › 1. Recap", "Recap of the previous session."),
    _chunk("section__c01_syl_sec05__00", "syllabus_section", "c01_syl", ["c01_syl_sec05"], [],
           "Course 1 · Syllabus › ATTENDANCE", "Students must attend at least 80% of the sessions."),
]


def answer(question: str, history: list[dict] | None = None, filters: Filters | None = None, cfg=None) -> AnswerResult:
    start = time.perf_counter()
    rag = True if cfg is None else bool(cfg.generation.get("rag", True))  # b0_no_rag.yaml sets rag to false
    out_of_scope = any(w in question.lower() for w in _OUT_OF_SCOPE_WORDS)
    rewritten = question if not history else f"{question} (rewritten with the previous question)"

    if not rag:  # the bare LLM: no search, no sources, and it invents a confident answer
        text = "The final exam is usually worth 50% of the grade and takes place in the last week of the course."
        hits, route, refused = [], RouteDecision(label="none", prob=0.0, source="none"), False
    elif out_of_scope:  # nothing relevant found: the system refuses instead of inventing
        text = REFUSAL
        hits, route, refused = [], RouteDecision(label="out_of_scope", prob=0.91, source="classifier"), True
    else:  # a normal cited answer with 5 sources
        hits = [
            Hit(chunk=c, rank=i + 1, score_dense=round(0.82 - 0.06 * i, 3), score_bm25=round(12.5 - 1.7 * i, 2),
                score_rrf=round(0.0325 - 0.0021 * i, 4), score_rerank=round(0.91 - 0.15 * i, 3), label=f"S{i + 1}")
            for i, c in enumerate(_SOURCES)
        ]
        text = ("This is a fake answer for designing the app. The final exam is worth 40% of the grade [S1], "
                "and the process in Session 4 has four steps, from input to output [S3]. "
                "Attendance must be at least 80% [S5].")
        route, refused = RouteDecision(label="course_admin", prob=0.93, source="classifier"), False

    total = (time.perf_counter() - start) * 1000
    timings = {"rewrite": 0.0 if not history else 310.0, "route": 4.0, "retrieve": 120.0 if rag else 0.0,
               "rerank": 380.0 if rag else 0.0, "generate": 5200.0, "total": round(total + 6014.0, 1)}
    return AnswerResult(
        question=question,
        rewritten_question=rewritten,
        route=route,
        filters_used=filters or Filters(),
        context_hits=hits,
        answer_text=text,
        citations=[Citation(label=h.label, chunk_id=h.chunk.chunk_id, covers=h.chunk.covers) for h in hits],
        refused=refused,
        timings_ms=timings,
        config_name=getattr(cfg, "name", "fake"),
        kb_sig="fake0000",
    )


def answer_stream(question: str, history: list[dict] | None = None, filters: Filters | None = None, cfg=None) -> Iterator[str | AnswerResult]:
    result = answer(question, history, filters, cfg)
    for word in result.answer_text.split(" "):  # yields the answer word by word, like the real LLM
        time.sleep(0.02)  # a small pause, so the app shows the text appearing
        yield word + " "
    yield result  # the last item is the full result, with the sources and the timings
