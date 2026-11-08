"""Reciprocal rank fusion of several result lists.

Owner: Sergio.
"""
from __future__ import annotations


def rrf(result_lists: list[list], k: int = 60) -> list:
    """Merge by chunk_id with score = sum of 1 / (k + rank)."""
    raise NotImplementedError("rrf: not built yet")
