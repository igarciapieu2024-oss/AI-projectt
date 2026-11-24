# Merge the result lists of the dense search and the BM25 search into one list with Reciprocal Rank Fusion
# (RRF). _build_hits() in src/generation/answer.py calls it, and the merged list then goes to the reranker.
# RRF only uses the position of each chunk in each list, not its score, because the two searches give scores
# on different scales (cosine similarity and BM25 points) that can't be added together.
from __future__ import annotations

# replace() gives us a copy of a Hit with a few fields changed.
from dataclasses import replace

# The format of a search result.
from src.schemas import Hit


# Each chunk gets 1 / (k + position) from every list it appears in, and the sums are sorted from highest to
# lowest. A chunk found by both searches adds two terms, so it moves up. k (rrf_k, 60 in base.yaml) is the
# usual RRF value: it keeps the difference between neighbouring positions small, so one list can't dominate
# the other.
def rrf(result_lists: list[list[Hit]], k: int = 60) -> list[Hit]:
    # k has to be a whole number of at least 1 (True and False are rejected too).
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive integer; got {k!r}")

    # One Hit per chunk, and its RRF sum.
    fused: dict[str, Hit] = {}
    scores: dict[str, float] = {}
    for results in result_lists:
        # A chunk that appears twice in the same list only counts once, at its best position.
        seen: set[str] = set()
        # The position in the list starts at 1, as in the RRF formula.
        for rank, hit in enumerate(results, start=1):
            chunk_id = hit.chunk.chunk_id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            # Add this list's term to the chunk's RRF sum.
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
            # The first time we see a chunk, we keep its Hit as it is.
            if chunk_id not in fused:
                fused[chunk_id] = hit
            else:
                # Found by both searches: we keep the dense score from one and the BM25 score from the other,
                # so the merged Hit carries both (the app shows them).
                previous = fused[chunk_id]
                fused[chunk_id] = replace(
                    previous,
                    score_dense=_max_score(previous.score_dense, hit.score_dense),
                    score_bm25=_max_score(previous.score_bm25, hit.score_bm25),
                    score_rerank=_max_score(previous.score_rerank, hit.score_rerank),
                )

    # Sort by the RRF sum; chunks with the same sum are ordered by id, so the order is always the same.
    ordered = sorted(
        fused.values(),
        key=lambda hit: (-scores[hit.chunk.chunk_id], hit.chunk.chunk_id),
    )
    # Number the merged list from 1 and store each chunk's RRF score.
    return [
        replace(hit, rank=rank, score_rrf=scores[hit.chunk.chunk_id])
        for rank, hit in enumerate(ordered, start=1)
    ]


# The larger of two scores, where None means "this search didn't give a score".
def _max_score(first: float | None, second: float | None) -> float | None:
    # If one of them is missing, we keep the other one.
    if first is None:
        return second
    if second is None:
        return first
    return max(first, second)
