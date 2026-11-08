"""Prompt templates v1 and v2.

Owner: Sergio.
"""
from __future__ import annotations


def build_messages(question: str, hits: list, cfg) -> list[dict]:
    """Return the chat messages for the LLM."""
    raise NotImplementedError("build_messages: not built yet")
