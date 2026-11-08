"""Cross-encoder reranker on the fused top hits.

Owner: Sergio.
"""
from __future__ import annotations


def rerank(query: str, hits: list, top_k: int, cfg) -> list:
    """Return the top_k hits by score_rerank."""
    raise NotImplementedError("rerank: not built yet")
