"""The answer function used by the app and the evaluation.

Owner: Sergio.
"""
from __future__ import annotations


def answer(question: str, history: list[dict] | None = None, filters=None, cfg=None):
    """Return an AnswerResult (see src/schemas.py)."""
    raise NotImplementedError("answer: not built yet")

def answer_stream(question: str, history: list[dict] | None = None, filters=None, cfg=None):
    """Yield text pieces as they arrive, then the final AnswerResult."""
    raise NotImplementedError("answer_stream: not built yet")
