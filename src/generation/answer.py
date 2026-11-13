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
from src.generation import rewrite, selfcheck, table_qa
from src.generation.llm import chat
from src.generation.prompts import build_messages
from src.retrieval import bm25, dense, fusion, rerank
from src.retrieval.router import route as route_question
from src.schemas import AnswerResult, Citation, Filters, Hit, REFUSAL, RouteDecision

_CITATION = re.compile(r"[\[(]S(\d+)[\])]", re.IGNORECASE)
_COMPLETE_CITED_SENTENCE = re.compile(
    r"(?:\s*\[S\d+\])*[.!?](?:\s*\[S\d+\])*(?=\s|$)",
    re.IGNORECASE,
)
_COMPLETE_SENTENCE = re.compile(r"[.!?](?=\s|$)")


def _trim_to_last_complete_sentence(answer_text: str, rag_enabled: bool) -> str:
    """Drop a length-truncated tail that does not end in a complete sentence."""
    sentence_pattern = _COMPLETE_CITED_SENTENCE if rag_enabled else _COMPLETE_SENTENCE
    completed = list(sentence_pattern.finditer(answer_text))
    if not completed:
        return ""
    return answer_text[:completed[-1].end()].rstrip()


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
) -> tuple[
    str, str, Config, list[Hit], list[dict[str, str]] | None, RouteDecision,
    float, float, float, float, float, bool,
]:
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")

    cfg = cfg or load_config("configs/final.yaml")
    started = time.perf_counter()
    rewritten_question = question
    rag_enabled = cfg.generation.get("rag", True)
    rewrite_ms = 0.0
    translation_enabled = rag_enabled and cfg.generation.get("translate_query", False)
    if (history and cfg.generation.get("rewrite", False)) or translation_enabled:
        rewrite_started = time.perf_counter()
        translate_to_english = translation_enabled and rewrite.detect_language(question) != "en"
        if history and cfg.generation.get("rewrite", False):
            if translate_to_english:
                rewritten_question = rewrite.rewrite(
                    question,
                    history,
                    cfg,
                    translate_to_english=True,
                )
            else:
                rewritten_question = rewrite.rewrite(question, history, cfg)
        elif translate_to_english:
            rewritten_question = rewrite.translate_to_english(question, cfg)
        rewrite_ms = (time.perf_counter() - rewrite_started) * 1000

    route_ms = 0.0
    route_decision = RouteDecision(label="none", prob=0.0, source="none")
    if rag_enabled:
        route_started = time.perf_counter()
        route_decision = route_question(rewritten_question, cfg)
        route_ms = (time.perf_counter() - route_started) * 1000

    if rag_enabled and route_decision.label == "out_of_scope":
        return (
            question, rewritten_question, cfg, [], None, route_decision, started,
            rewrite_ms, route_ms, 0.0, 0.0, rag_enabled,
        )

    retrieval_started = time.perf_counter()
    table_hits = False
    hits = []
    if rag_enabled:
        if route_decision.label == "cross_course":
            hits = table_qa.answer_from_table(rewritten_question, cfg)
            table_hits = bool(hits)
        if not hits:
            hits = _build_hits(rewritten_question, cfg, filters)
    retrieve_ms = (time.perf_counter() - retrieval_started) * 1000
    rerank_ms = 0.0
    if rag_enabled and hits and not table_hits:
        top_k = _top_k(cfg)
        if cfg.retrieval.get("rerank", False):
            rerank_started = time.perf_counter()
            hits = rerank.rerank(rewritten_question, hits, top_k, cfg)
            rerank_ms = (time.perf_counter() - rerank_started) * 1000
        else:
            hits = hits[:top_k]
        is_weak = cfg.generation.get("selfcheck", False) and selfcheck.is_weak(hits, cfg)
        hits = [replace(hit, rank=rank, label=f"S{rank}") for rank, hit in enumerate(hits, start=1)]
    elif table_hits:
        is_weak = False
        hits = [replace(hit, rank=rank, label=f"S{rank}") for rank, hit in enumerate(hits, start=1)]
    else:
        is_weak = False
    messages = (
        build_messages(question, hits, cfg, history)
        if not rag_enabled or (hits and not is_weak)
        else None
    )
    return (
        question, rewritten_question, cfg, hits, messages, route_decision, started,
        rewrite_ms, route_ms, retrieve_ms, rerank_ms, rag_enabled,
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
    route_decision: RouteDecision,
    filters: Filters | None,
    cfg: Config,
    hits: list[Hit],
    answer_text: str,
    started: float,
    rewrite_ms: float,
    route_ms: float,
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
        route=route_decision,
        filters_used=filters or Filters(),
        context_hits=hits,
        answer_text=answer_text,
        citations=citations,
        refused=rag_enabled and answer_text == REFUSAL,
        timings_ms={
            "rewrite": round(rewrite_ms, 1),
            "route": round(route_ms, 1),
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
    (
        question, rewritten_question, cfg, hits, messages, route_decision, started,
        rewrite_ms, route_ms, retrieve_ms, rerank_ms, rag_enabled,
    ) = _prepare_answer(
        question, history, filters, cfg
    )
    if rag_enabled and messages is None:
        return _make_result(
            question, rewritten_question, route_decision, filters, cfg, hits, REFUSAL, started,
            rewrite_ms, route_ms, retrieve_ms, rerank_ms, 0.0, rag_enabled,
        )
    if messages is None:
        raise RuntimeError("answer prompt was not constructed")

    generation_started = time.perf_counter()
    response = chat(messages, cfg)
    if not isinstance(response, str):
        raise TypeError("chat() must return text when stream=False")
    answer_text = str(response)
    if getattr(response, "done_reason", None) == "length":
        answer_text = _trim_to_last_complete_sentence(answer_text, rag_enabled)
    if not answer_text.strip() and getattr(response, "done_reason", None) != "length":
        raise ValueError("chat() returned empty answer text")
    generate_ms = (time.perf_counter() - generation_started) * 1000
    return _make_result(
        question, rewritten_question, route_decision, filters, cfg, hits, answer_text, started,
        rewrite_ms, route_ms, retrieve_ms, rerank_ms, generate_ms, rag_enabled,
    )


def answer_stream(
    question: str,
    history: list[dict] | None = None,
    filters: Filters | None = None,
    cfg: Config | None = None,
) -> Iterator[str | AnswerResult]:
    """Yield generated text pieces and then the validated final result."""
    (
        question, rewritten_question, cfg, hits, messages, route_decision, started,
        rewrite_ms, route_ms, retrieve_ms, rerank_ms, rag_enabled,
    ) = _prepare_answer(
        question, history, filters, cfg
    )
    if rag_enabled and messages is None:
        result = _make_result(
            question, rewritten_question, route_decision, filters, cfg, hits, REFUSAL, started,
            rewrite_ms, route_ms, retrieve_ms, rerank_ms, 0.0, rag_enabled,
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
    pending = ""
    for piece in response:
        pieces.append(piece)
        pending += piece
        sentence_pattern = _COMPLETE_CITED_SENTENCE if rag_enabled else _COMPLETE_SENTENCE
        completed = list(sentence_pattern.finditer(pending))
        if completed:
            end = completed[-1].end()
            yield pending[:end]
            pending = pending[end:]
    done_reason = getattr(response, "done_reason", None)
    if done_reason != "length" and pending:
        yield pending
    answer_text = "".join(pieces)
    if done_reason == "length":
        answer_text = _trim_to_last_complete_sentence(answer_text, rag_enabled)
    if not answer_text.strip() and done_reason != "length":
        raise ValueError("chat() returned no answer text while streaming")
    generate_ms = (time.perf_counter() - generation_started) * 1000
    result = _make_result(
        question, rewritten_question, route_decision, filters, cfg, hits, answer_text, started,
        rewrite_ms, route_ms, retrieve_ms, rerank_ms, generate_ms, rag_enabled,
    )
    if not answer_text.strip():
        yield result.answer_text
    yield result
