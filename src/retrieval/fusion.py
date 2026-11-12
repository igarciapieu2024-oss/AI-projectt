"""Reciprocal rank fusion of several result lists.

Owner: Sergio.
"""
from __future__ import annotations

from dataclasses import replace

from src.schemas import Hit


def rrf(result_lists: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Merge ranked lists by chunk ID using reciprocal-rank fusion."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive integer; got {k!r}")

    fused: dict[str, Hit] = {}
    scores: dict[str, float] = {}
    for results in result_lists:
        seen: set[str] = set()
        for rank, hit in enumerate(results, start=1):
            chunk_id = hit.chunk.chunk_id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
            if chunk_id not in fused:
                fused[chunk_id] = hit
            else:
                previous = fused[chunk_id]
                fused[chunk_id] = replace(
                    previous,
                    score_dense=_max_score(previous.score_dense, hit.score_dense),
                    score_bm25=_max_score(previous.score_bm25, hit.score_bm25),
                    score_rerank=_max_score(previous.score_rerank, hit.score_rerank),
                )

    ordered = sorted(
        fused.values(),
        key=lambda hit: (-scores[hit.chunk.chunk_id], hit.chunk.chunk_id),
    )
    return [
        replace(hit, rank=rank, score_rrf=scores[hit.chunk.chunk_id])
        for rank, hit in enumerate(ordered, start=1)
    ]


def _max_score(first: float | None, second: float | None) -> float | None:
    if first is None:
        return second
    if second is None:
        return first
    return max(first, second)
