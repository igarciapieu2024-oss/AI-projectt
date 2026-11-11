"""Question-answering API used by the app and evaluation.

Owner: Sergio.

Retrieves evidence from ChromaDB and BM25, then generates a cited answer locally.
"""
from __future__ import annotations

import re
import time
from dataclasses import replace
from typing import Iterator

from src.config import Config, load_config
from src.generation.llm import chat
from src.generation.prompts import build_messages
from src.generation import rewrite, selfcheck
from src.retrieval import bm25, dense, fusion, rerank
from src.schemas import AnswerResult, Citation, Filters, Hit, REFUSAL, RouteDecision

_CITATION = re.compile(r"[\[(]S(\d+)[\])]", re.IGNORECASE)


def _top_k(cfg: Config) -> int:
    top_k = cfg.retrieval.get("top_k", 5)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError(f"retrieval.top_k must be a positive integer; got {top_k!r}")
    return top_k


def _build_hits(question: str, cfg: Config, filters: Filters | None) -> list[Hit]:
    top_k = _top_k(cfg)
    mode = cfg.retrieval.get("mode", "hybrid")
    if mode not in {"dense", "hybrid"}:
        raise ValueError(f"unsupported retrieval.mode: {mode!r}")
    dense_k = cfg.retrieval.get("k_dense", top_k)
    if not isinstance(dense_k, int) or isinstance(dense_k, bool) or dense_k < 1:
        raise ValueError(f"retrieval.k_dense must be a positive integer; got {dense_k!r}")

    dense_hits = dense.search(question, cfg, filters, dense_k)
    if mode == "dense":
        return dense_hits

    bm25_k = cfg.retrieval.get("k_bm25", top_k)
    if not isinstance(bm25_k, int) or isinstance(bm25_k, bool) or bm25_k < 1:
        raise ValueError(f"retrieval.k_bm25 must be a positive integer; got {bm25_k!r}")
    bm25_hits = bm25.search(question, cfg, filters, bm25_k)
    rrf_k = cfg.retrieval.get("rrf_k", 60)
    if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k < 1:
        raise ValueError(f"retrieval.rrf_k must be a positive integer; got {rrf_k!r}")
    return fusion.rrf([dense_hits, bm25_hits], k=rrf_k)


def _prepare_answer(
    question: str,
    history: list[dict] | None,
    filters: Filters | None,
    cfg: Config | None,
) -> tuple[str, str, Config, list[Hit], list[dict[str, str]] | None, float, float, float, float, bool]:
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")

    cfg = cfg or load_config("configs/final.yaml")
    started = time.perf_counter()
    rewritten_question = question
    rag_enabled = cfg.generation.get("rag", True)
    rewrite_ms = 0.0
    if history and cfg.generation.get("rewrite", False):
        rewrite_started = time.perf_counter()
        rewritten_question = rewrite.rewrite(question, history, cfg)
        rewrite_ms = (time.perf_counter() - rewrite_started) * 1000

    retrieval_started = time.perf_counter()
    hits = _build_hits(rewritten_question, cfg, filters) if rag_enabled else []
    retrieve_ms = (time.perf_counter() - retrieval_started) * 1000
    rerank_ms = 0.0
    if rag_enabled and hits:
        top_k = _top_k(cfg)
        if cfg.retrieval.get("rerank", False):
            rerank_started = time.perf_counter()
            hits = rerank.rerank(rewritten_question, hits, top_k, cfg)
            rerank_ms = (time.perf_counter() - rerank_started) * 1000
        else:
            hits = hits[:top_k]
        if cfg.generation.get("selfcheck", False) and selfcheck.is_weak(hits, cfg):
            hits = []
        hits = [replace(hit, rank=rank, label=f"S{rank}") for rank, hit in enumerate(hits, start=1)]
    messages = build_messages(question, hits, cfg, history) if not rag_enabled or hits else None
    return (
        question, rewritten_question, cfg, hits, messages, started,
        rewrite_ms, retrieve_ms, rerank_ms, rag_enabled,
    )


def _validated_answer(answer_text: str, hits: list[Hit], rag_enabled: bool) -> tuple[str, list[Hit]]:
    if not rag_enabled:
        return answer_text.strip(), []

    hits_by_label = {hit.label: hit for hit in hits}
    used_labels: list[str] = []
    invalid_citation = False

    def normalize_citation(match: re.Match[str]) -> str:
        nonlocal invalid_citation
        label = f"S{match.group(1)}"
        if label not in hits_by_label:
            invalid_citation = True
            return match.group(0)
        if label not in used_labels:
            used_labels.append(label)
        return f"[{label}]"

    answer_text = _CITATION.sub(normalize_citation, answer_text).strip()
    if invalid_citation or not used_labels or not answer_text:
        return REFUSAL, []
    return answer_text, [hits_by_label[label] for label in used_labels]


def _make_result(
    question: str,
    rewritten_question: str,
    filters: Filters | None,
    cfg: Config,
    hits: list[Hit],
    answer_text: str,
    started: float,
    rewrite_ms: float,
    retrieve_ms: float,
    rerank_ms: float,
    generate_ms: float,
    rag_enabled: bool,
) -> AnswerResult:
    answer_text, cited_hits = _validated_answer(answer_text, hits, rag_enabled)
    citations = [
        Citation(label=hit.label, chunk_id=hit.chunk.chunk_id, covers=hit.chunk.covers)
        for hit in cited_hits
    ]
    return AnswerResult(
        question=question,
        rewritten_question=rewritten_question,
        route=RouteDecision(label="none", prob=0.0, source="none"),
        filters_used=filters or Filters(),
        context_hits=hits,
        answer_text=answer_text,
        citations=citations,
        refused=rag_enabled and answer_text == REFUSAL,
        timings_ms={
            "rewrite": round(rewrite_ms, 1),
            "route": 0.0,
            "retrieve": round(retrieve_ms, 1),
            "rerank": round(rerank_ms, 1),
            "generate": round(generate_ms, 1),
            "total": round((time.perf_counter() - started) * 1000, 1),
        },
        config_name=cfg.name,
        kb_sig=cfg.kb_sig,
    )


def answer(
    question: str,
    history: list[dict] | None = None,
    filters: Filters | None = None,
    cfg: Config | None = None,
) -> AnswerResult:
    """Generate an answer from retrieved evidence, or run the no-RAG baseline."""
    question, rewritten_question, cfg, hits, messages, started, rewrite_ms, retrieve_ms, rerank_ms, rag_enabled = _prepare_answer(
        question, history, filters, cfg
    )
    if rag_enabled and not hits:
        return _make_result(
            question, rewritten_question, filters, cfg, hits, REFUSAL, started,
            rewrite_ms, retrieve_ms, rerank_ms, 0.0, rag_enabled,
        )
    if messages is None:
        raise RuntimeError("answer prompt was not constructed")

    generation_started = time.perf_counter()
    response = chat(messages, cfg)
    if not isinstance(response, str):
        raise TypeError("chat() must return text when stream=False")
    if not response.strip():
        raise ValueError("chat() returned empty answer text")
    generate_ms = (time.perf_counter() - generation_started) * 1000
    return _make_result(
        question, rewritten_question, filters, cfg, hits, response, started,
        rewrite_ms, retrieve_ms, rerank_ms, generate_ms, rag_enabled,
    )


def answer_stream(
    question: str,
    history: list[dict] | None = None,
    filters: Filters | None = None,
    cfg: Config | None = None,
) -> Iterator[str | AnswerResult]:
    """Yield generated text pieces and then the validated final result."""
    question, rewritten_question, cfg, hits, messages, started, rewrite_ms, retrieve_ms, rerank_ms, rag_enabled = _prepare_answer(
        question, history, filters, cfg
    )
    if rag_enabled and not hits:
        result = _make_result(
            question, rewritten_question, filters, cfg, hits, REFUSAL, started,
            rewrite_ms, retrieve_ms, rerank_ms, 0.0, rag_enabled,
        )
        yield result.answer_text
        yield result
        return
    if messages is None:
        raise RuntimeError("answer prompt was not constructed")

    generation_started = time.perf_counter()
    response = chat(messages, cfg, stream=True)
    if isinstance(response, str):
        raise TypeError("chat() must return an iterator when stream=True")
    pieces: list[str] = []
    for piece in response:
        pieces.append(piece)
        yield piece
    if not "".join(pieces).strip():
        raise ValueError("chat() returned no answer text while streaming")
    generate_ms = (time.perf_counter() - generation_started) * 1000
    result = _make_result(
        question, rewritten_question, filters, cfg, hits, "".join(pieces), started,
        rewrite_ms, retrieve_ms, rerank_ms, generate_ms, rag_enabled,
    )
    yield result
