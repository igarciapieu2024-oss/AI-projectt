"""Question-answering API used by the app and evaluation.

Owner: Sergio.

Retrieval currently uses a local chunk-ranking fallback until the shared index
interface is ready.
"""
from __future__ import annotations

import math
import re
import time
from typing import Iterator

from src.config import Config, PATHS, load_config
from src.generation.llm import chat
from src.generation.prompts import build_messages
from src.schemas import AnswerResult, Citation, Chunk, Filters, Hit, REFUSAL, RouteDecision
from src.utils.io import read_jsonl

_TOKEN = re.compile(r"\w+%?", re.UNICODE)
_CITATION = re.compile(r"[\[(]S(\d+)[\])]", re.IGNORECASE)


def _load_chunks(cfg: Config) -> list[Chunk]:
    path = PATHS["processed"] / f"chunks__{cfg.kb_sig}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"Knowledge-base chunks not found: {path}. "
            f"Build them with `python -m src.pipeline build --config configs/{cfg.name}.yaml`."
        )
    return read_jsonl(path, Chunk)


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN.findall(text.casefold()))


def _filtered_chunks(chunks: list[Chunk], filters: Filters | None) -> list[Chunk]:
    if filters is None:
        return chunks

    return [
        chunk
        for chunk in chunks
        if (filters.course_code is None or chunk.course_code == filters.course_code)
        and (not filters.sessions or any(session in chunk.sessions for session in filters.sessions))
    ]


def _top_k(cfg: Config) -> int:
    top_k = cfg.retrieval.get("top_k", 5)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError(f"retrieval.top_k must be a positive integer; got {top_k!r}")
    return top_k


def _build_hits(question: str, cfg: Config, filters: Filters | None) -> list[Hit]:
    top_k = _top_k(cfg)
    all_chunks = _load_chunks(cfg)
    chunks = _filtered_chunks(all_chunks, filters)
    query_tokens = _tokenize(question)
    if not query_tokens or not chunks:
        return []

    tokenized_corpus = [
        (chunk, _tokenize(chunk.embed_text or chunk.text))
        for chunk in all_chunks
    ]
    document_frequency: dict[str, int] = {}
    for _, tokens in tokenized_corpus:
        for token in tokens:
            document_frequency[token] = document_frequency.get(token, 0) + 1

    n_documents = len(tokenized_corpus)
    scored: list[tuple[float, Chunk]] = []
    for chunk in chunks:
        if not chunk.text.strip():
            continue
        tokens = _tokenize(chunk.embed_text or chunk.text)
        matches = query_tokens & tokens
        if not matches:
            continue

        # Corpus-wide IDF weights informative terms without a hand-maintained word list.
        score = sum(
            math.log1p(n_documents / document_frequency[token])
            for token in matches
        )
        scored.append((score, chunk))

    scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
    return [
        Hit(chunk=chunk, rank=rank, label=f"S{rank}")
        for rank, (_, chunk) in enumerate(scored[:top_k], start=1)
    ]


def _prepare_answer(
    question: str,
    history: list[dict] | None,
    filters: Filters | None,
    cfg: Config | None,
) -> tuple[str, Config, list[Hit], list[dict[str, str]] | None, float, float, bool]:
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")

    cfg = cfg or load_config("configs/final.yaml")
    started = time.perf_counter()
    rag_enabled = cfg.generation.get("rag", True)
    retrieval_started = time.perf_counter()
    hits = _build_hits(question, cfg, filters) if rag_enabled else []
    retrieve_ms = (time.perf_counter() - retrieval_started) * 1000
    messages = build_messages(question, hits, cfg, history) if not rag_enabled or hits else None
    return question, cfg, hits, messages, started, retrieve_ms, rag_enabled


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
    filters: Filters | None,
    cfg: Config,
    hits: list[Hit],
    answer_text: str,
    started: float,
    retrieve_ms: float,
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
        rewritten_question=question,
        route=RouteDecision(label="none", prob=0.0, source="none"),
        filters_used=filters or Filters(),
        context_hits=hits,
        answer_text=answer_text,
        citations=citations,
        refused=rag_enabled and answer_text == REFUSAL,
        timings_ms={
            "rewrite": 0.0,
            "route": 0.0,
            "retrieve": round(retrieve_ms, 1),
            "rerank": 0.0,
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
    question, cfg, hits, messages, started, retrieve_ms, rag_enabled = _prepare_answer(
        question, history, filters, cfg
    )
    if rag_enabled and not hits:
        return _make_result(
            question, filters, cfg, hits, REFUSAL, started, retrieve_ms, 0.0, rag_enabled
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
        question, filters, cfg, hits, response, started, retrieve_ms, generate_ms, rag_enabled
    )


def answer_stream(
    question: str,
    history: list[dict] | None = None,
    filters: Filters | None = None,
    cfg: Config | None = None,
) -> Iterator[str | AnswerResult]:
    """Yield generated text pieces and then the validated final result."""
    question, cfg, hits, messages, started, retrieve_ms, rag_enabled = _prepare_answer(
        question, history, filters, cfg
    )
    if rag_enabled and not hits:
        result = _make_result(
            question, filters, cfg, hits, REFUSAL, started, retrieve_ms, 0.0, rag_enabled
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
        question, filters, cfg, hits, "".join(pieces), started, retrieve_ms, generate_ms, rag_enabled
    )
    yield result
