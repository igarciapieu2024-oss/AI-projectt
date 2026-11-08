"""Fake answer() with the same inputs and outputs as the real one (owner: Sergio).

The app and the evaluation use it until the real engine exists:
    USE_FAKE=1 streamlit run app.py
It never searches or calls an LLM; it always returns the same two sources.
"""
from __future__ import annotations

import time
from typing import Iterator

from src.schemas import AnswerResult, Chunk, Citation, Filters, Hit, RouteDecision

_SOURCES = [
    Chunk(
        chunk_id="section__c01_syl_sec04__00", kb_sig="fake0000", strategy="section",
        doc_type="syllabus_section", course_code="c01", sessions=[], source_id="c01_syl",
        covers=["c01_syl_sec04"], parent_id="c01_syl_oevaluation", outline_path="Syllabus › EVALUATION",
        text="Final exam: 40%. Group project: 30%. Class participation: 20%. Quizzes: 10%.",
        embed_text="Syllabus › EVALUATION\nFinal exam: 40%. Group project: 30%.",
    ),
    Chunk(
        chunk_id="slide__c01_s04_p017__00", kb_sig="fake0000", strategy="slide",
        doc_type="slide", course_code="c01", sessions=[4], source_id="c01_s04",
        covers=["c01_s04_p017"], parent_id="c01_s04_o2-1-subtopic", outline_path="Session 4 › 2. Topic › 2.1 Subtopic",
        text="Example slide text. [Image: diagram] Example image description.",
        embed_text="Session 4 › 2. Topic › 2.1 Subtopic\nExample slide text.", has_image_text=True,
    ),
]


def answer(question: str, history: list[dict] | None = None, filters: Filters | None = None, cfg=None) -> AnswerResult:
    start = time.perf_counter()
    hits = [
        Hit(chunk=c, rank=i + 1, score_dense=0.8 - 0.1 * i, score_bm25=12.0 - i, score_rrf=0.03 - 0.001 * i,
            score_rerank=0.9 - 0.2 * i, label=f"S{i + 1}")
        for i, c in enumerate(_SOURCES)
    ]
    text = "This is a fake answer for testing the app: the final exam is worth 40% [S1], and the topic is covered in Session 4 [S2]."
    total = (time.perf_counter() - start) * 1000
    return AnswerResult(
        question=question,
        rewritten_question=question,
        route=RouteDecision(label="course_admin", prob=0.93, source="classifier"),
        filters_used=filters or Filters(),
        context_hits=hits,
        answer_text=text,
        citations=[Citation(label=h.label, chunk_id=h.chunk.chunk_id, covers=h.chunk.covers) for h in hits],
        refused=False,
        timings_ms={"rewrite": 0.0, "route": 1.0, "retrieve": 2.0, "rerank": 1.0, "generate": 5.0, "total": round(total + 9.0, 1)},
        config_name=getattr(cfg, "name", "fake"),
        kb_sig="fake0000",
    )


def answer_stream(question: str, history: list[dict] | None = None, filters: Filters | None = None, cfg=None) -> Iterator[str | AnswerResult]:
    result = answer(question, history, filters, cfg)
    for word in result.answer_text.split(" "):
        yield word + " "
    yield result
