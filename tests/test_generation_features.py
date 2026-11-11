from types import SimpleNamespace

import pytest

import src.generation.answer as answer_module
import src.generation.rewrite as rewrite_module
import src.generation.selfcheck as selfcheck_module
from src.retrieval.filters import extract_filters
from src.schemas import Chunk, Filters, Hit


def _chunk():
    return Chunk(
        chunk_id="chunk_a",
        kb_sig="test",
        strategy="slide",
        doc_type="slide",
        course_code="c01",
        sessions=[2],
        source_id="c01_s02",
        covers=["c01_s02_p001"],
        parent_id="chunk_a_parent",
        outline_path="Session 2",
        text="Evidence text.",
        embed_text="Evidence text.",
    )


def test_extract_filters_matches_course_names_codes_and_sessions():
    courses = [
        {"course_code": "mc", "course_name": "Management Control"},
        {"course_code": "macro", "course_name": "Macroeconomics"},
    ]

    filters = extract_filters(
        "In MANAGEMENT CONTROL, compare sessions 2 and session 11.",
        courses,
    )

    assert filters == Filters(course_code="mc", sessions=[2, 11])
    assert extract_filters("Question about mc, session 4", courses).course_code == "mc"


def test_extract_filters_does_not_guess_between_multiple_courses():
    courses = [
        {"course_code": "c01", "course_name": "Course One"},
        {"course_code": "c02", "course_name": "Course Two"},
    ]

    assert extract_filters("Compare Course One and Course Two", courses).course_code is None


def test_rewrite_uses_history_and_returns_only_standalone_question(monkeypatch):
    calls = []

    def fake_chat(messages, cfg):
        calls.append(messages)
        return "What is the minimum grade to pass Management Control?"

    monkeypatch.setattr(rewrite_module, "chat", fake_chat)
    history = [
        {"role": "user", "content": "What is the exam worth?"},
        {"role": "assistant", "content": "The final exam is worth 40% [S1]."},
    ]

    result = rewrite_module.rewrite("And what is the minimum grade?", history, object())

    assert result == "What is the minimum grade to pass Management Control?"
    assert calls[0][-1]["content"] == "And what is the minimum grade?"
    assert calls[0][1:] == history + [{"role": "user", "content": "And what is the minimum grade?"}]


def test_rewrite_skips_model_without_history_and_rejects_empty_response(monkeypatch):
    monkeypatch.setattr(
        rewrite_module,
        "chat",
        lambda *args: pytest.fail("rewrite without history should not call the model"),
    )
    assert rewrite_module.rewrite("Question", [], object()) == "Question"
    with pytest.raises(ValueError, match="history entries"):
        rewrite_module.rewrite("Question", [{"role": "system", "content": "bad"}], object())

    monkeypatch.setattr(rewrite_module, "chat", lambda *args: " ")
    with pytest.raises(ValueError, match="empty rewritten"):
        rewrite_module.rewrite("Question", [{"role": "user", "content": "Earlier"}], object())


def test_selfcheck_uses_best_reranker_score_and_requires_scores():
    cfg = SimpleNamespace(generation={"selfcheck_threshold": 0.2})
    assert selfcheck_module.is_weak([], cfg)
    assert not selfcheck_module.is_weak([Hit(chunk=_chunk(), rank=1, score_rerank=0.4)], cfg)
    assert selfcheck_module.is_weak([Hit(chunk=_chunk(), rank=1, score_rerank=0.1)], cfg)
    with pytest.raises(ValueError, match="requires reranked hits"):
        selfcheck_module.is_weak([Hit(chunk=_chunk(), rank=1)], cfg)


def test_answer_rewrites_before_retrieval_and_refuses_weak_evidence(monkeypatch):
    calls = []
    hit = Hit(chunk=_chunk(), rank=1, score_rerank=0.1)
    monkeypatch.setattr(
        answer_module.rewrite,
        "rewrite",
        lambda question, history, cfg: "Standalone rewritten question",
    )
    monkeypatch.setattr(
        answer_module,
        "_build_hits",
        lambda query, cfg, filters: calls.append(query) or [hit],
    )
    monkeypatch.setattr(
        answer_module.rerank,
        "rerank",
        lambda query, hits, top_k, cfg: hits,
    )
    monkeypatch.setattr(
        answer_module,
        "chat",
        lambda *args, **kwargs: pytest.fail("weak evidence should refuse without generation"),
    )
    cfg = SimpleNamespace(
        name="test",
        kb_sig="test",
        retrieval={"mode": "dense", "top_k": 1, "rerank": True},
        generation={
            "rag": True,
            "rewrite": True,
            "selfcheck": True,
            "selfcheck_threshold": 0.2,
            "prompt_version": "v2",
        },
    )

    result = answer_module.answer(
        "And what about that?",
        history=[{"role": "user", "content": "Previous question"}],
        cfg=cfg,
    )

    assert calls == ["Standalone rewritten question"]
    assert result.rewritten_question == "Standalone rewritten question"
    assert result.refused
    assert result.answer_text == "This isn't covered in the course material I have."
    assert result.context_hits == []
