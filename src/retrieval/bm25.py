"""Keyword search (BM25), using tokenize() from src.kb.build_indexes.

Owner: Sergio.
"""
from __future__ import annotations

import pickle

from src.config import bm25_path, chunks_path
from src.kb.build_indexes import tokenize
from src.schemas import Chunk, Filters, Hit
from src.utils.io import read_jsonl


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive integer; got {k!r}")


def _matches_filters(chunk, filters: Filters | None) -> bool:
    return filters is None or (
        (filters.course_code is None or chunk.course_code == filters.course_code)
        and (not filters.sessions or any(session in chunk.sessions for session in filters.sessions))
    )


def search(query: str, cfg, filters: Filters | None, k: int) -> list[Hit]:
    """Return the top BM25 chunks, with exact course/session filtering."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    _validate_k(k)
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    index_path = bm25_path(cfg.kb_sig)
    if not index_path.is_file():
        raise FileNotFoundError(
            f"BM25 index not found: {index_path}. "
            f"Build indexes with `python -m src.pipeline build --config configs/{cfg.name}.yaml`."
        )
    with index_path.open("rb") as stream:
        index = pickle.load(stream)
    expected_keys = {"kb_sig", "chunk_ids", "tokenized_corpus", "bm25"}
    if not isinstance(index, dict) or set(index) != expected_keys:
        raise ValueError(f"BM25 index at {index_path} has an invalid format")
    if index["kb_sig"] != cfg.kb_sig:
        raise ValueError(
            f"BM25 index kb_sig mismatch: expected {cfg.kb_sig}, found {index['kb_sig']}"
        )
    chunk_ids = index["chunk_ids"]
    tokenized_corpus = index["tokenized_corpus"]
    if not isinstance(chunk_ids, list) or not isinstance(tokenized_corpus, list):
        raise ValueError(f"BM25 index at {index_path} has invalid corpus data")
    if len(chunk_ids) != len(tokenized_corpus):
        raise ValueError(f"BM25 index at {index_path} has mismatched corpus lengths")

    chunk_file = chunks_path(cfg.kb_sig)
    if not chunk_file.is_file():
        raise FileNotFoundError(f"Chunk file not found for kb_sig={cfg.kb_sig}: {chunk_file}")
    chunks = {chunk.chunk_id: chunk for chunk in read_jsonl(chunk_file, Chunk)}
    try:
        scores = index["bm25"].get_scores(query_tokens)
    except (AttributeError, TypeError) as exc:
        raise ValueError(f"BM25 index at {index_path} has no compatible scorer") from exc
    if len(scores) != len(chunk_ids):
        raise ValueError(f"BM25 scorer at {index_path} returned an invalid score count")

    scored = []
    for chunk_id, score in zip(chunk_ids, scores):
        chunk = chunks.get(chunk_id)
        if chunk is None:
            raise KeyError(f"BM25 chunk {chunk_id!r} is missing from {chunk_file}")
        if score > 0 and _matches_filters(chunk, filters) and chunk.text.strip():
            scored.append((float(score), chunk))
    scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
    return [
        Hit(chunk=chunk, rank=rank, score_bm25=score)
        for rank, (score, chunk) in enumerate(scored[:k], start=1)
    ]
