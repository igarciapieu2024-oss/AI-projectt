"""Follow-up question -> standalone question, using the chat history.

Owner: Sergio.
"""
from __future__ import annotations

import re
from typing import Any

from src.generation.llm import chat


def detect_language(question: str) -> str: # Return the ISO language code most likely used by a short question
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be empty")
    from langid import classify

    language, _ = classify(question.strip())
    return language


def translate_to_english(question: str, cfg) -> str: # Translate a non-English search query to English without answering it.
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


# 1. Words that point back to the conversation. A question without any of them is read on its own:
#    rewriting it with the history added the previous course to new questions
#    ("Which courses have the highest final exam weight?" became a Management Control question).
_FOLLOW_UP = re.compile(
    r"\b(it|its|that|this|these|those|they|them|their|he|she|him|her|his|there|same|previous|above|"
    r"former|latter|eso|esa|ese|esto|ello|ella|ellos|ellas|anterior|mismo|misma)\b"
    r"|^\s*[¿¡]?\s*(and|what about|how about|y|también)\b",
    re.IGNORECASE,
)


def looks_like_follow_up(question: str) -> bool: # true when the question refers to the conversation or is too short to stand alone
    return bool(_FOLLOW_UP.search(question)) or len(question.split()) <= 3


def rewrite( # rewrite a follow-up as a standalone question without answering it
    question: str,
    history: list[dict[str, Any]],
    cfg,
    *,
    translate_to_english: bool = False,
) -> str:
    
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    if not history or not looks_like_follow_up(question):
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
    # 1. With generation.rewrite_guard, a rewrite that is not a question is thrown away. The model sometimes
    #    answered the follow-up instead of rewriting it, and once invented the answer ("10 dollars per kilogram"),
    #    which the router then sent out of scope. The previous user question plus the new one is used instead.
    if getattr(cfg, "generation", {}).get("rewrite_guard", False) and not looks_like_question(rewritten):
        return _join_with_previous(question, history)
    return rewritten


_CITATION = re.compile(r"\[S\d+\]")


def looks_like_question(text: str) -> bool: # usable rewrite is one line, ends with a question mark and carries no source labels
    text = text.strip()
    return bool(text) and "\n" not in text and text.endswith("?") and not _CITATION.search(text)


def _join_with_previous(question: str, history: list[dict[str, Any]]) -> str: # the last user question and the follow-up together, so the search still sees the topic
    previous = [item["content"].strip() for item in history if item.get("role") == "user"]
    if not previous:
        return question
    return f"{previous[-1]} {question}"


def translate_to_english_query(question: str, cfg) -> str: # translate only non-English text, preserving English queries unchanged
    if detect_language(question) == "en":
        return question
    return translate_to_english(question, cfg)


def _query_token_limit(cfg) -> int: # keep short query transformations on their own small generation budget
    return getattr(cfg, "generation", {}).get("query_num_predict", 96)
