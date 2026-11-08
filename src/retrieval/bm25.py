"""Keyword search (BM25), using tokenize() from src.kb.build_indexes.

Owner: Sergio.
"""
from __future__ import annotations


def search(query: str, cfg, filters, k: int) -> list:
    """Return list[Hit] with score_bm25."""
    raise NotImplementedError("search: not built yet")
