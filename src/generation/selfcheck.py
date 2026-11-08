"""Refuse when the retrieved material is too weak (after one retry).

Owner: Sergio.
"""
from __future__ import annotations


def is_weak(hits: list, cfg) -> bool:
    """True when the best reranker score is under selfcheck_threshold."""
    raise NotImplementedError("is_weak: not built yet")
