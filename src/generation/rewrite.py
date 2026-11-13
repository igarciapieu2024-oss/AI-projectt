"""Follow-up question -> standalone question, using the chat history.

Owner: Sergio.
"""
from __future__ import annotations

from typing import Any

from src.generation.llm import chat


def detect_language(question: str) -> str:
    """Return the ISO language code most likely used by a short question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be empty")
    from langid import classify

    language, _ = classify(question.strip())
    return language


def translate_to_english(question: str, cfg) -> str:
    """Translate a non-English search query to English without answering it."""
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    result = chat(
        [
            {
                "role": "system",
                "content": (
                    "Translate the user's question into English for searching course "
                    "materials. Preserve course names, technical terms, and numbers. "
                    "Do not answer the question or add facts. Output only the English "
                    "translation."
                ),
            },
            {"role": "user", "content": question},
        ],
        cfg,
        num_predict=_query_token_limit(cfg),
    )
    if not isinstance(result, str):
        raise TypeError("chat() must return text when stream=False")
    translated = result.strip().strip("\"'")
    if not translated:
        raise ValueError("chat() returned an empty translated question")
    return translated


def rewrite(
    question: str,
    history: list[dict[str, Any]],
    cfg,
    *,
    translate_to_english: bool = False,
) -> str:
    """Rewrite a follow-up as a standalone question without answering it."""
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    if not history:
        return translate_to_english_query(question, cfg) if translate_to_english else question

    instruction = (
        "Rewrite the latest user question as a standalone question using the "
        "conversation only to resolve references. Do not answer it, add facts, "
        "or include explanations. Output only the rewritten question."
    )
    if translate_to_english:
        instruction = (
            "Rewrite the latest user question as a standalone question using the "
            "conversation only to resolve references, and translate it into English. "
            "Preserve course names, technical terms, and numbers. Do not answer it, "
            "add facts, or include explanations. Output only the English question."
        )
    messages = [{
        "role": "system",
        "content": instruction,
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
    result = chat(messages, cfg, num_predict=_query_token_limit(cfg))
    if not isinstance(result, str):
        raise TypeError("chat() must return text when stream=False")
    rewritten = result.strip().strip("\"'")
    if not rewritten:
        raise ValueError("chat() returned an empty rewritten question")
    return rewritten


def translate_to_english_query(question: str, cfg) -> str:
    """Translate only non-English text, preserving English queries unchanged."""
    if detect_language(question) == "en":
        return question
    return translate_to_english(question, cfg)


def _query_token_limit(cfg) -> int:
    """Keep short query transformations on their own small generation budget."""
    return getattr(cfg, "generation", {}).get("query_num_predict", 96)
