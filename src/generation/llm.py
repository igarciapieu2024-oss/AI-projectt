"""Ollama chat wrapper.

Owner: Sergio.
"""
from __future__ import annotations


def chat(messages: list[dict], cfg, stream: bool = False, json: bool = False):
    """Return the reply text, or an iterator of text pieces when stream=True."""
    raise NotImplementedError("chat: not built yet")
