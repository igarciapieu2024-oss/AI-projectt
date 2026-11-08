"""Chunks -> ChromaDB collection kb_<kb_sig> + BM25 index bm25__<kb_sig>.pkl.

Owner: Santiago.
"""
from __future__ import annotations


def tokenize(text: str) -> list[str]:
    """Lower-case words, numbers and % (shared with Sergio's BM25 search)."""
    raise NotImplementedError("tokenize: not built yet")

def build_indexes(cfg) -> int:
    """Load non-duplicate chunks into Chroma and BM25; return the indexed count."""
    raise NotImplementedError("build_indexes: not built yet")
