"""Follow-up question -> standalone question, using the chat history.

Owner: Sergio.
"""
from __future__ import annotations

from typing import Any

from src.generation.llm import chat


def rewrite(question: str, history: list[dict[str, Any]], cfg) -> str:
    """Rewrite a follow-up as a standalone question without answering it."""
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    if not history:
        return question

    messages = [{
        "role": "system",
        "content": (
            "Rewrite the latest user question as a standalone question using the "
            "conversation only to resolve references. Do not answer it, add facts, "
            "or include explanations. Output only the rewritten question."
        ),
    }]
    for item in history:
        if (
            not isinstance(item, dict)
            or item.get("role") not in {"user", "assistant"}
            or not isinstance(item.get("content"), str)
        ):
            raise ValueError("history entries must have user/assistant roles and string content")
        messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": question})
    result = chat(messages, cfg)
    if not isinstance(result, str):
        raise TypeError("chat() must return text when stream=False")
    rewritten = result.strip().strip("\"'")
    if not rewritten:
        raise ValueError("chat() returned an empty rewritten question")
    return rewritten
