# Tests of the steps around the answer: course filters, follow-up rewriting, translation and the self-check.
import math
from types import SimpleNamespace

import pytest

import src.generation.answer as answer_module
import src.generation.rewrite as rewrite_module
import src.generation.selfcheck as selfcheck_module
from src.retrieval.filters import extract_filters
from src.schemas import Chunk, Filters, Hit, RouteDecision


# A small fake chunk for the tests.
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


# Courses are found by name or code, and sessions by number.
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


# With two courses named, no course is chosen.
def test_extract_filters_does_not_guess_between_multiple_courses():
    courses = [
        {"course_code": "c01", "course_name": "Course One"},
        {"course_code": "c02", "course_name": "Course Two"},
    ]

    assert extract_filters("Compare Course One and Course Two", courses).course_code is None


# A follow-up is rewritten with the conversation, and only the question comes back.
def test_rewrite_uses_history_and_returns_only_standalone_question(monkeypatch):
    calls = []

    def fake_chat(messages, cfg, **kwargs):
        calls.append(messages)
        assert kwargs == {"num_predict": 96}
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


# A follow-up in another language is rewritten and translated in one call.
def test_rewrite_can_resolve_and_translate_a_follow_up(monkeypatch):
    calls = []

    def fake_chat(messages, cfg, **kwargs):
        calls.append(messages)
        assert kwargs == {"num_predict": 96}
        return "What is the attendance limit for Management Control?"

    monkeypatch.setattr(rewrite_module, "chat", fake_chat)
    history = [
        {"role": "user", "content": "¿Cuál es la asistencia mínima?"},
        {"role": "assistant", "content": "El curso exige un 80% de asistencia."},
    ]

    result = rewrite_module.rewrite(
        "¿Y en Management Control?",
        history,
        object(),
        translate_to_english=True,
    )

    assert result == "What is the attendance limit for Management Control?"
    assert "translate it into English" in calls[0][0]["content"]
    assert calls[0][-1]["content"] == "¿Y en Management Control?"


# Without a conversation the model isn't called, and an empty rewrite is an error.
def test_rewrite_skips_model_without_history_and_rejects_empty_response(monkeypatch):
    monkeypatch.setattr(
        rewrite_module,
        "chat",
        lambda *args: pytest.fail("rewrite without history should not call the model"),
    )
    assert rewrite_module.rewrite("Question", [], object()) == "Question"
    with pytest.raises(ValueError, match="history entries"):
        rewrite_module.rewrite("Question", [{"role": "system", "content": "bad"}], object())

    monkeypatch.setattr(rewrite_module, "chat", lambda *args, **kwargs: " ")
    with pytest.raises(ValueError, match="empty rewritten"):
        rewrite_module.rewrite("Question", [{"role": "user", "content": "Earlier"}], object())


# A Spanish question is detected as Spanish.
def test_language_detection_recognizes_spanish():
    assert rewrite_module.detect_language("¿Quién da la asignatura de Strategies?") == "es"


# English questions aren't translated; Spanish ones are.
def test_translation_helper_skips_english_and_translates_spanish(monkeypatch):
    calls = []

    def fake_translate(question, cfg):
        calls.append(question)
        return "Who teaches Strategies?"

    monkeypatch.setattr(rewrite_module, "translate_to_english", fake_translate)

    assert rewrite_module.translate_to_english_query("Who teaches Strategies?", object()) == (
        "Who teaches Strategies?"
    )
    assert rewrite_module.translate_to_english_query(
        "¿Quién da la asignatura de Strategies?", object()
    ) == "Who teaches Strategies?"
    assert calls == ["¿Quién da la asignatura de Strategies?"]


# The search uses the translated question, but the model answers the original one.
def test_answer_translates_search_query_but_keeps_original_question_for_generation(monkeypatch):
    original_question = "¿Cuánto vale el examen final de Management Control?"
    search_question = "What is the final exam worth in Management Control?"
    hit = Hit(chunk=_chunk(), rank=1)
    searched = []
    generated = []
    cfg = SimpleNamespace(
        name="test",
        kb_sig="test",
        retrieval={"mode": "dense", "top_k": 1, "rerank": False},
        generation={
            "rag": True,
            "translate_query": True,
            "prompt_version": "v2",
        },
    )
    monkeypatch.setattr(rewrite_module, "detect_language", lambda question: "es")
    monkeypatch.setattr(
        rewrite_module,
        "translate_to_english",
        lambda question, config: search_question,
    )
    monkeypatch.setattr(
        answer_module,
        "route_question",
        lambda question, config: RouteDecision(label="none", prob=0.0, source="none"),
    )
    monkeypatch.setattr(
        answer_module,
        "_build_hits",
        lambda question, config, filters: searched.append(question) or [hit],
    )

    def fake_chat(messages, config, stream=False):
        generated.append(messages)
        return "El examen final vale el 40%. [S1]"

    monkeypatch.setattr(answer_module, "chat", fake_chat)

    result = answer_module.answer(original_question, cfg=cfg)

    assert searched == [search_question]
    assert result.rewritten_question == search_question
    assert original_question in generated[0][-1]["content"]
    assert "same language as the user's question" in generated[0][0]["content"]
    assert result.answer_text == "El examen final vale el 40%. [S1]"


# The self-check looks at the best reranker score, and needs the scores to exist.
def test_selfcheck_uses_best_reranker_score_and_requires_scores():
    cfg = SimpleNamespace(generation={"selfcheck_threshold": 0.2})
    assert selfcheck_module.is_weak([], cfg)
    assert not selfcheck_module.is_weak([Hit(chunk=_chunk(), rank=1, score_rerank=0.4)], cfg)
    assert selfcheck_module.is_weak([Hit(chunk=_chunk(), rank=1, score_rerank=0.1)], cfg)
    with pytest.raises(ValueError, match="requires reranked hits"):
        selfcheck_module.is_weak([Hit(chunk=_chunk(), rank=1)], cfg)


# The threshold is compared with the 0-to-1 confidence, not with the raw score.
def test_selfcheck_threshold_uses_probability_scale_for_negative_raw_logits():
    cfg = SimpleNamespace(generation={"selfcheck_threshold": 0.2})
    raw_logit = -0.5
    confidence = 1 / (1 + math.exp(-raw_logit))
    assert confidence > 0.2
    assert not selfcheck_module.is_weak(
        [Hit(chunk=_chunk(), rank=1, score_rerank=confidence)],
        cfg,
    )


# The prompt keeps the question in its original language.
def test_rag_answer_prompt_preserves_the_question_language():
    from src.generation.prompts import build_messages

    cfg = SimpleNamespace(generation={"rag": True, "prompt_version": "v2"})
    hit = Hit(chunk=_chunk(), rank=1, label="S1")

    messages = build_messages("¿Cuánto vale el examen final?", [hit], cfg)

    assert "same language as the user's question" in messages[0]["content"]
    assert "1 to 3 sentences" in messages[0]["content"]


# The question is rewritten before searching, and weak sources lead to a refusal.
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
    assert len(result.context_hits) == 1
    assert result.context_hits[0].chunk.chunk_id == hit.chunk.chunk_id
    assert result.context_hits[0].label == "S1"


# A question that isn't a follow-up stays as it is, even with a conversation.
def test_rewrite_keeps_a_standalone_question_even_with_history(monkeypatch):
    from src.generation import rewrite as rewrite_module

    # 1. a question that stands on its own is not rewritten, so the previous course is not added to it
    monkeypatch.setattr(rewrite_module, "chat", lambda *args, **kwargs: pytest.fail("the model should not be called"))
    history = [
        {"role": "user", "content": "What is the final exam worth in Management Control?"},
        {"role": "assistant", "content": "40% [S1]"},
    ]
    question = "Which courses have the highest final exam weight?"
    assert rewrite_module.rewrite(question, history, SimpleNamespace(generation={})) == question
    # 2. a question that points back to the conversation is still rewritten
    assert rewrite_module.looks_like_follow_up("And who teaches it?")
    assert rewrite_module.looks_like_follow_up("¿Y la asistencia mínima?")


# With rewrite_guard, a rewrite that is an answer is replaced by the previous question plus the new one.
def test_rewrite_guard_replaces_an_answer_with_the_previous_question(monkeypatch):
    # 1. the model answered the follow-up (and invented the price); with the guard the two questions are joined
    monkeypatch.setattr(
        rewrite_module,
        "chat",
        lambda messages, cfg, **kwargs: "The chefs bought the material at 10 dollars per kilogram [S3].",
    )
    history = [
        {"role": "user", "content": "In the Singadeli example, what is the standard cost of the material?"},
        {"role": "assistant", "content": "11.8 dollars per kilogram [S1]."},
    ]
    cfg = SimpleNamespace(generation={"rewrite_guard": True})
    result = rewrite_module.rewrite("And at what price did they buy it?", history, cfg)
    assert result == (
        "In the Singadeli example, what is the standard cost of the material? And at what price did they buy it?"
    )


# A real question passes the guard, and the guard is off by default.
def test_rewrite_guard_keeps_a_real_question_and_is_off_by_default(monkeypatch):
    # 2. a proper rewrite passes the guard; without the option an answer-like rewrite is kept as before
    history = [{"role": "user", "content": "Who teaches Strategy?"}]
    monkeypatch.setattr(rewrite_module, "chat", lambda *a, **k: "Who teaches the Strategy course this term?")
    cfg = SimpleNamespace(generation={"rewrite_guard": True})
    assert rewrite_module.rewrite("And this term?", history, cfg) == "Who teaches the Strategy course this term?"
    monkeypatch.setattr(rewrite_module, "chat", lambda *a, **k: "The professor is X [S1].")
    assert rewrite_module.rewrite("And this term?", history, SimpleNamespace(generation={})) == "The professor is X [S1]."


# What counts as a question: one line, ending in "?", with no source labels.
def test_looks_like_question():
    # 3. one line, a question mark at the end and no source labels
    assert rewrite_module.looks_like_question("What is the passing grade in OB?")
    assert not rewrite_module.looks_like_question("The passing grade is 5.")
    assert not rewrite_module.looks_like_question("What is it [S1]?")
    assert not rewrite_module.looks_like_question("What is it?\nIt is 5.")
