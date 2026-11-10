
"""LLM judge for answer correctness, claims, and citations.

Uses the shared Ollama chat interface and the configured judge model.

Owner: Elena.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any, Callable

from src.generation.llm import chat


def check_citations(
    answer_text: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check whether citation labels refer to supplied sources."""
    used = set(re.findall(r"\[(S\d+)\]", answer_text))
    available = {
        source["label"]
        for source in sources
        if source.get("label")
    }

    return {
        "valid_labels": sorted(used & available),
        "invalid_labels": sorted(used - available),
        "citation_labels_valid": not bool(used - available),
    }


def build_judge_messages(
    answer_text: str,
    reference_answer: str,
    sources: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build a prompt for evidence-based answer evaluation."""
    source_text = json.dumps(
        sources, ensure_ascii=False, indent=2
    )

    system = """
You are an independent evaluator of a course assistant.

Evaluate the answer using ONLY the supplied reference answer
and retrieved source texts.

Instructions:
1. Give correctness a score from 1 to 5.
   1 = incorrect, 3 = partially correct, 5 = correct.
2. Identify factual claims made in the answer.
3. Mark a claim supported only when the provided evidence
   actually supports it.
4. Do not assume a citation makes a claim supported.
5. Do not invent missing evidence.
6. Return ONLY a JSON object with this structure:

{
  "correctness": 1,
  "claims": [
    {
      "text": "A factual claim",
      "supported": true
    }
  ],
  "reasoning": "Brief explanation"
}

The claims list must include the factual claims being evaluated.
Use JSON booleans true and false.
"""

    user = (
        f"REFERENCE ANSWER:\n{reference_answer}\n\n"
        f"ASSISTANT ANSWER:\n{answer_text}\n\n"
        f"RETRIEVED SOURCES:\n{source_text}"
    )

    return [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user},
    ]


def judge_answer(
    answer_text: str,
    reference_answer: str,
    sources: list[dict[str, Any]],
    cfg: Any,
    chat_fn: Callable | None = None,
) -> dict[str, Any]:
    """Evaluate one answer with the configured LLM judge.

    chat_fn allows deterministic tests without running Ollama.
    """
    judge_model = cfg.eval.get("judge_model")

    if not judge_model:
        raise ValueError(
            "Missing eval.judge_model in configuration."
        )

    # Keep the original generation configuration unchanged.
    judge_cfg = copy.copy(cfg)
    judge_cfg.generation = {
        **cfg.generation,
        "llm_model": judge_model,
        "temperature": 0.0,
    }

    messages = build_judge_messages(
        answer_text,
        reference_answer,
        sources,
    )

    call_chat = chat_fn if chat_fn is not None else chat

    raw = call_chat(
        messages,
        judge_cfg,
        stream=False,
        json=True,
    )

    if not isinstance(raw, str):
        raise ValueError("LLM judge must return a JSON string.")

    try:
        # Allow a response wrapped in a Markdown JSON code fence.
        cleaned = raw.strip()
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            cleaned,
        )
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(
            "LLM judge returned invalid JSON."
        ) from exc

    if not isinstance(data, dict):
        raise ValueError("LLM judge response must be an object.")

    correctness = data.get("correctness")
    claims = data.get("claims")
    reasoning = data.get("reasoning")

    if (
        type(correctness) is not int
        or not 1 <= correctness <= 5
    ):
        raise ValueError("Correctness must be an integer from 1 to 5.")

    if not isinstance(claims, list):
        raise ValueError("Claims must be a list.")

    if not isinstance(reasoning, str):
        raise ValueError("Reasoning must be a string.")

    for claim in claims:
        if (
            not isinstance(claim, dict)
            or not isinstance(claim.get("text"), str)
            or type(claim.get("supported")) is not bool
        ):
            raise ValueError("Invalid claim structure.")

    unsupported = [
        claim["text"]
        for claim in claims
        if not claim["supported"]
    ]

    hallucination_rate = (
        len(unsupported) / len(claims)
        if claims else None
    )

    return {
        "judge_model": judge_model,
        "correctness": correctness,
        "claims": claims,
        "n_claims": len(claims),
        "unsupported_claims": unsupported,
        "n_unsupported_claims": len(unsupported),
        "hallucination_rate": hallucination_rate,
        "citation_check": check_citations(
            answer_text, sources
        ),
        "reasoning": reasoning,
    }
