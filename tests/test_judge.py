
"""Unit tests for the LLM evaluation judge.

Owner: Elena.
"""
import json

import pytest

from eval.judge import (
    build_judge_messages,
    check_citations,
    judge_answer,
)
from src.config import load_config


@pytest.fixture
def cfg():
    """Load the project's evaluation configuration."""
    return load_config("configs/b1_v1.yaml")


@pytest.fixture
def sources():
    """Example retrieved evidence."""
    return [
        {
            "label": "S1",
            "text": "The final exam is worth 40%.",
            "covers": ["c01_syl_sec04"],
        }
    ]


def fake_chat(messages, cfg, stream=False, json=False):
    """Simulate a valid Ollama judge response."""
    assert cfg.generation["llm_model"] == "qwen2.5:7b"
    assert json is True

    return (
        '{"correctness": 5, "claims": ['
        '{"text": "The final exam is worth 40%", '
        '"supported": true}], '
        '"reasoning": "The statement matches the source."}'
    )


def test_judge_success(cfg, sources):
    """The judge parses a valid response and computes metrics."""
    result = judge_answer(
        answer_text="The final exam is worth 40% [S1].",
        reference_answer="The final exam is worth 40%.",
        sources=sources,
        cfg=cfg,
        chat_fn=fake_chat,
    )

    assert result["correctness"] == 5
    assert result["n_claims"] == 1
    assert result["n_unsupported_claims"] == 0
    assert result["hallucination_rate"] == 0.0
    assert result["citation_check"]["citation_labels_valid"] is True


def test_unsupported_claims(cfg, sources):
    """Unsupported claims contribute to hallucination rate."""
    def fake_unsupported(messages, cfg, stream=False, json=False):
        return (
            '{"correctness": 2, "claims": ['
            '{"text": "Exam is worth 40%", "supported": true},'
            '{"text": "Exam is online", "supported": false}],'
            '"reasoning": "One claim has no supporting evidence."}'
        )

    result = judge_answer(
        answer_text="The exam is worth 40% and is online [S1].",
        reference_answer="The exam is worth 40%.",
        sources=sources,
        cfg=cfg,
        chat_fn=fake_unsupported,
    )

    assert result["correctness"] == 2
    assert result["n_claims"] == 2
    assert result["n_unsupported_claims"] == 1
    assert result["hallucination_rate"] == 0.5


def test_invalid_citation(sources):
    """Detect citations that do not refer to supplied sources."""
    result = check_citations(
        "The final exam is worth 40% [S9].",
        sources,
    )

    assert result["citation_labels_valid"] is False
    assert result["invalid_labels"] == ["S9"]


def test_invalid_json(cfg, sources):
    """Invalid LLM output raises a clear error."""
    def fake_invalid(messages, cfg, stream=False, json=False):
        return "not valid json"

    with pytest.raises(ValueError, match="invalid JSON"):
        judge_answer(
            answer_text="Example answer",
            reference_answer="Example reference",
            sources=sources,
            cfg=cfg,
            chat_fn=fake_invalid,
        )


def test_invalid_correctness_score(cfg, sources):
    """Correctness must be an integer between 1 and 5."""
    def fake_bad_score(messages, cfg, stream=False, json=False):
        return (
            '{"correctness": 7, "claims": [], '
            '"reasoning": "Invalid score."}'
        )

    with pytest.raises(ValueError, match="Correctness"):
        judge_answer(
            answer_text="Example answer",
            reference_answer="Example reference",
            sources=sources,
            cfg=cfg,
            chat_fn=fake_bad_score,
        )


def test_no_claims(cfg, sources):
    """No factual claims means hallucination rate is undefined."""
    def fake_empty(messages, cfg, stream=False, json=False):
        return (
            '{"correctness": 5, "claims": [], '
            '"reasoning": "No factual claims."}'
        )

    result = judge_answer(
        answer_text="Thank you.",
        reference_answer="Thank you.",
        sources=sources,
        cfg=cfg,
        chat_fn=fake_empty,
    )

    assert result["n_claims"] == 0
    assert result["hallucination_rate"] is None


def test_judge_messages(sources):
    """The prompt contains the reference and retrieved evidence."""
    messages = build_judge_messages(
        answer_text="The final exam is worth 40%.",
        reference_answer="The final exam is worth 40%.",
        sources=sources,
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "The final exam is worth 40%" in messages[1]["content"]
    assert "c01_syl_sec04" in messages[1]["content"]
