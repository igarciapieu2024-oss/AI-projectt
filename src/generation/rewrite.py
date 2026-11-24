# Prepare the question before the search: turn a follow-up into a standalone question using the chat
# history, and translate a question that is not in English. Called by _prepare_answer() in
# src/generation/answer.py: rewrite() only when there is a conversation history and generation.rewrite
# is true, and the translation only when generation.translate_query is true (both true in base.yaml).
# The result is used by the router and the search; the LLM still receives the original question and
# the history when it writes the answer.
from __future__ import annotations

# Regular expressions, to recognise a follow-up and a source label.
import re
from typing import Any

# chat() sends the messages to the local LLM (llm.py).
from src.generation.llm import chat


# ----------------------------------------------------------------------
# Translation (for questions that are not in English)
# All the course material is in English, so a question in another language is translated before it is
# searched. With English questions these functions make no call to the LLM.

# Return the language code of the question ("en" for English), detected with the langid library.
def detect_language(question: str) -> str:
    # An empty question has no language to detect.
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be empty")
    # Imported here so the library is only loaded when a language has to be detected.
    from langid import classify

    # classify() returns the language and a score; only the language is needed.
    language, _ = classify(question.strip())
    return language


# Translate a question into English with the LLM, only to search with it; the model is told not to answer.
def translate_to_english(question: str, cfg) -> str:
    # An empty question can't be translated.
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    # The instruction keeps course names, technical terms and numbers as they are, because the search
    # needs them exactly, and forbids answering or adding facts.
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
        # A short limit, since a translated question is only a few words long.
        num_predict=_query_token_limit(cfg),
    )
    # chat() returns text when it is not streaming; anything else is an error.
    if not isinstance(result, str):
        raise TypeError("chat() must return text when stream=False")
    # Remove spaces and quotation marks the model sometimes puts around the translation.
    translated = result.strip().strip("\"'")
    if not translated:
        raise ValueError("chat() returned an empty translated question")
    return translated


# ----------------------------------------------------------------------
# Follow-up questions

# Words that point back to the conversation (pronouns such as "it" or "that", "same", "previous", and the
# same words in Spanish), or a question that starts with "and", "what about" or "how about". A question
# without any of them is read on its own: rewriting every question with the history added the course of
# the previous question to new, unrelated questions.
_FOLLOW_UP = re.compile(
    r"\b(it|its|that|this|these|those|they|them|their|he|she|him|her|his|there|same|previous|above|"
    r"former|latter|eso|esa|ese|esto|ello|ella|ellos|ellas|anterior|mismo|misma)\b"
    r"|^\s*[¿¡]?\s*(and|what about|how about|y|también)\b",
    re.IGNORECASE,
)


# True when the question refers to the conversation, or has three words or fewer and is therefore too
# short to be understood on its own.
def looks_like_follow_up(question: str) -> bool:
    return bool(_FOLLOW_UP.search(question)) or len(question.split()) <= 3


# Rewrite a follow-up as a standalone question with the LLM, using the conversation only to know what
# the question refers to. The model is told not to answer it. translate_to_english also translates it
# in the same call, for follow-ups that are not in English.
def rewrite(
    question: str,
    history: list[dict[str, Any]],
    cfg,
    *,
    translate_to_english: bool = False,
) -> str:
    # An empty question can't be rewritten.
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    # No conversation, or not a follow-up: the question is kept as it is (only translated if needed).
    if not history or not looks_like_follow_up(question):
        return translate_to_english_query(question, cfg) if translate_to_english else question

    # The instruction for the LLM: rewrite, do not answer, add nothing, output only the question.
    instruction = (
        "Rewrite the latest user question as a standalone question using the "
        "conversation only to resolve references. Do not answer it, add facts, "
        "or include explanations. Output only the rewritten question."
    )
    # The same instruction, also asking for an English translation.
    if translate_to_english:
        instruction = (
            "Rewrite the latest user question as a standalone question using the "
            "conversation only to resolve references, and translate it into English. "
            "Preserve course names, technical terms, and numbers. Do not answer it, "
            "add facts, or include explanations. Output only the English question."
        )
    # The messages sent to the LLM: the instruction, the whole conversation and the new question last.
    messages = [{
        "role": "system",
        "content": instruction,
    }]
    for item in history:
        # Every message of the history must be a user or assistant message with text.
        if (
            not isinstance(item, dict)
            or item.get("role") not in {"user", "assistant"}
            or not isinstance(item.get("content"), str)
        ):
            raise ValueError("history entries must have user/assistant roles and string content")
        messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": question})
    # A short token limit (query_num_predict), since the result is only one question.
    result = chat(messages, cfg, num_predict=_query_token_limit(cfg))
    # chat() returns text when it isn't streaming; anything else is an error.
    if not isinstance(result, str):
        raise TypeError("chat() must return text when stream=False")
    # Remove spaces and quotation marks around the rewritten question.
    rewritten = result.strip().strip("\"'")
    if not rewritten:
        raise ValueError("chat() returned an empty rewritten question")
    # With generation.rewrite_guard (final_system.yaml), a rewrite that is not a question is thrown away.
    # The model sometimes answered the follow-up instead of rewriting it, and once invented the answer,
    # which the router then sent out of scope. The previous user question plus the new one is used instead.
    if getattr(cfg, "generation", {}).get("rewrite_guard", False) and not looks_like_question(rewritten):
        return _join_with_previous(question, history)
    return rewritten


# A source label such as [S1]: if it appears, the model wrote an answer, not a question.
_CITATION = re.compile(r"\[S\d+\]")


# True when the rewrite can be used as a question: one line, ending with a question mark and without
# source labels. Used by the rewrite_guard check above.
def looks_like_question(text: str) -> bool:
    # One line, ending in a question mark, and with no source labels.
    text = text.strip()
    return bool(text) and "\n" not in text and text.endswith("?") and not _CITATION.search(text)


# The last user question and the follow-up joined together, used when the rewrite is discarded, so the
# search still sees the topic of the conversation.
def _join_with_previous(question: str, history: list[dict[str, Any]]) -> str:
    # The questions the student asked before, in order.
    previous = [item["content"].strip() for item in history if item.get("role") == "user"]
    # Without a previous user question, the follow-up is searched as it is.
    if not previous:
        return question
    return f"{previous[-1]} {question}"


# Translate the question only if it is not in English; an English question is returned unchanged,
# without calling the LLM.
def translate_to_english_query(question: str, cfg) -> str:
    if detect_language(question) == "en":
        return question
    return translate_to_english(question, cfg)


# The token limit for rewriting and translating questions (generation.query_num_predict, 96 in base.yaml).
# It is kept separate from the limit of the answers (num_predict, 400), because a question is much shorter.
def _query_token_limit(cfg) -> int:
    return getattr(cfg, "generation", {}).get("query_num_predict", 96)
