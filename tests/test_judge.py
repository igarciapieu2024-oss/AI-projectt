
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
    return load_config("configs/baselines/b1_first_version.yaml")


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


# "Not in the material" counts as a correct refusal only for unanswerable questions.
def test_semantic_refusal_counts_only_for_unanswerable_questions():
    from eval.judge import judge_one, says_not_in_material

    answer = "There is no information provided about the teaching assistant for Time Series Analysis. [S1]"
    # 1. an unanswerable question answered "not in the material" is a correct refusal, without the judge model
    unanswerable = {"answerable": "0", "result": {"refused": False, "answer_text": answer}}
    assert judge_one(unanswerable, model="unused")["score"] == 1.0
    # 2. an unanswerable question answered with a made-up fact is still wrong
    invented = {"answerable": "0", "result": {"refused": False, "answer_text": "The final quiz is worth 35%. [S4]"}}
    assert judge_one(invented, model="unused")["score"] == 0.0
    assert says_not_in_material("The exact date is not specified in the provided sources.")
    assert says_not_in_material("The complete list of questions is not explicitly provided in the sources.")
    assert not says_not_in_material("The final exam is worth 40% of the final grade. [S1]")


# Rubric v2 uses its own instructions, which judge what the question asks.
def test_rubric_v2_judges_what_the_question_asks(monkeypatch):
    import sys
    import types

    from eval import judge

    # 1. v2 sends the question-focused rubric to the judge model; v1 stays the default
    sent = {}

    def fake_chat(model, messages, format, options):
        sent["system"] = messages[0]["content"]
        return {"message": {"content": '{"correct": true, "score": 1, "reason": "ok"}'}}

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(chat=fake_chat))
    row = {"answerable": "1", "question": "What share are quizzes?", "reference_answer": "25%; best four count",
           "key_facts": "25%", "result": {"refused": False, "answer_text": "25% [S1]"}}
    assert judge.judge_one(row, "m", "v2")["score"] == 1.0
    assert "what the QUESTION asks" in sent["system"]
    judge.judge_one(row, "m")
    assert sent["system"] == judge.SYSTEM


# The agreement counts when the judge is stricter and when it is more lenient.
def test_judge_agreement_counts_strict_and_lenient_decisions():
    # 1. one answer the judge failed although it was right, one it passed although it was half right
    from eval.judge_agreement import agreement

    reference = {("r", "Q1"): 1.0, ("r", "Q2"): 0.5, ("r", "Q3"): 0.0}
    judged = {("r", "Q1"): 0.0, ("r", "Q2"): 1.0, ("r", "Q3"): 0.0}
    result = agreement(reference, judged)
    assert result["compared"] == 3
    assert result["judge_too_strict"] == 1 and result["judge_too_lenient"] == 1
    assert result["same_correct_decision"] == round(1 / 3, 4)


# Several gold sources are joined into one text for the judge.
def test_gold_source_text_joins_several_sources(monkeypatch):
    # 2. a cross-course question has two gold sections; both reach the v3 judge
    import eval.judge as judge_module

    monkeypatch.setattr(judge_module, "_GOLD_TEXTS", {"a_syl_sec08": "Final exam 30%", "b_syl_sec08": "Final exam 20%"})
    text = judge_module.gold_source_text("a_syl_sec08;b_syl_sec08")
    assert "Final exam 30%" in text and "Final exam 20%" in text
