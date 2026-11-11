"""Refuse when the retrieved material is too weak.

Owner: Sergio.
"""
from __future__ import annotations

import math


def is_weak(hits: list, cfg) -> bool:
    """Compare normalized cross-encoder confidence to the configured threshold."""
    if not hits:
        return True
    threshold = cfg.generation.get("selfcheck_threshold", 0.2)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(f"generation.selfcheck_threshold must be numeric; got {threshold!r}")
    if not math.isfinite(threshold):
        raise ValueError("generation.selfcheck_threshold must be finite")
    scores = [hit.score_rerank for hit in hits]
    if any(score is None for score in scores):
        raise ValueError("self-check requires reranked hits with score_rerank values")
    return max(scores) < threshold
