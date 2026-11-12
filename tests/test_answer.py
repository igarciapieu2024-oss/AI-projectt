import re
from types import SimpleNamespace

import pytest

import src.generation.answer as answer_module
from src.schemas import AnswerResult, Chunk, Filters, Hit, RouteDecision


def _chunk(
    chunk_id: str,
    text: str,
    *,
    course_code: str = "c01",
    sessions: list[int] | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        kb_sig="test",
        strategy="slide",
        doc_type="slide",
        course_code=course_code,
        sessions=sessions or [],
        source_id=f"{course_code}_s01",
        covers=[f"{course_code}_s01_p001"],
        parent_id=f"{chunk_id}_parent",
        outline_path="",
        text=text,
        embed_text=text,
    )


def _config(top_k: int = 5):
    return SimpleNamespace(
        name="test",
        kb_sig="test",
        retrieval={
            "mode": "hybrid",
            "top_k": top_k,
            "k_dense": 20,
            "k_bm25": 20,
            "rrf_k": 60,
            "rerank": False,
        },
        generation={"rag": True, "prompt_version": "v2"},
    )


def _mock_hits(chunks, question, cfg, filters=None):
    query_tokens = set(re.findall(r"\w+%?", question.casefold()))
    candidates = []
    for chunk in chunks:
        if filters and filters.course_code and chunk.course_code != filters.course_code:
            continue
        if filters and filters.sessions and not any(session in chunk.sessions for session in filters.sessions):
            continue
        tokens = set(re.findall(r"\w+%?", chunk.text.casefold()))
        score = len(query_tokens & tokens)
        if score:
            candidates.append((score, chunk))
    candidates.sort(key=lambda item: (-item[0], item[1].chunk_id))
    return [
        Hit(chunk=chunk, rank=rank)
        for rank, (_, chunk) in enumerate(candidates[:cfg.retrieval["top_k"]], start=1)
    ]


@pytest.fixture(autouse=True)
def _mock_chat(monkeypatch):
    calls = []

    def fake_chat(messages, cfg, stream=False):
        calls.append((messages, cfg, stream))
        if stream:
            return iter(["Exam weight is 40 percent. [S1]"])
        return "Exam weight is 40 percent. [S1]"

    monkeypatch.setattr(answer_module, "chat", fake_chat)
    return calls


def test_answer_respects_top_k_and_cites_used_evidence(monkeypatch):
    chunks = [
        _chunk("chunk_a", "Exam weight is 40 percent."),
        _chunk("chunk_b", "Exam weight is 30 percent."),
        _chunk("chunk_c", "Attendance is required."),
        _chunk("chunk_d", "The course covers operations."),
        _chunk("chunk_e", "This lecture explains forecasting."),
    ]
    monkeypatch.setattr(answer_module, "_build_hits", lambda q, cfg, filters: _mock_hits(chunks, q, cfg, filters))

    result = answer_module.answer("What is the exam weight?", cfg=_config(top_k=1))

    assert len(result.context_hits) == 1
    assert result.answer_text == "Exam weight is 40 percent. [S1]"
    assert [(citation.label, citation.chunk_id) for citation in result.citations] == [
        ("S1", "chunk_a")
    ]
    assert result.route.label == "none"
    assert result.route.prob == 0
    assert result.timings_ms["total"] >= result.timings_ms["retrieve"]


def test_citations_only_include_sources_used_by_the_model(monkeypatch):
    chunks = [
        _chunk("chunk_a", "Exam weight is 40 percent."),
        _chunk("chunk_b", "Exam weight is 30 percent."),
    ]
    monkeypatch.setattr(answer_module, "_build_hits", lambda q, cfg, filters: _mock_hits(chunks, q, cfg, filters))
    monkeypatch.setattr(
        answer_module,
        "chat",
        lambda messages, cfg, stream=False: "The exam weighs 40 percent [S1].",
    )

    result = answer_module.answer("Exam weight", cfg=_config(top_k=2))

    assert result.answer_text == "The exam weighs 40 percent [S1]."
    assert [citation.label for citation in result.citations] == ["S1"]
    assert len(result.context_hits) == 2


def test_answer_applies_course_and_session_filters(monkeypatch):
    chunks = [
        _chunk("wrong_course", "Capstone project.", course_code="c02", sessions=[2]),
        _chunk("wrong_session", "Capstone project.", sessions=[1]),
        _chunk("matching", "Capstone project.", sessions=[2]),
    ]
    monkeypatch.setattr(answer_module, "_build_hits", lambda q, cfg, filters: _mock_hits(chunks, q, cfg, filters))

    result = answer_module.answer(
        "Capstone project",
        filters=Filters(course_code="c01", sessions=[2]),
        cfg=_config(),
    )

    assert [hit.chunk.chunk_id for hit in result.context_hits] == ["matching"]


def test_no_matching_chunks_is_a_refusal(monkeypatch):
    chunks = [
        _chunk("unrelated", "Attendance is required.")
    ]
    monkeypatch.setattr(answer_module, "_build_hits", lambda q, cfg, filters: _mock_hits(chunks, q, cfg, filters))

    result = answer_module.answer("Explain quantum tunneling", cfg=_config())

    assert result.refused
    assert result.context_hits == []
    assert result.citations == []
    assert result.answer_text == "This isn't covered in the course material I have."


def test_invalid_question_and_top_k_are_rejected(monkeypatch):
    with pytest.raises(ValueError, match="must not be empty"):
        answer_module.answer("  ", cfg=_config())
    with pytest.raises(ValueError, match="positive integer"):
        answer_module.answer("Question", cfg=_config(top_k=0))


def test_no_rag_config_calls_model_without_retrieval(monkeypatch, _mock_chat):
    cfg = _config()
    cfg.generation = {"rag": False}
    messages_seen = []
    monkeypatch.setattr(
        answer_module,
        "_build_hits",
        lambda *args: pytest.fail("no-RAG mode must not retrieve course chunks"),
    )

    def fake_chat(messages, cfg):
        messages_seen.append(messages)
        return "A general-knowledge answer."

    monkeypatch.setattr(
        answer_module,
        "chat",
        fake_chat,
    )

    result = answer_module.answer("Question", cfg=cfg)

    assert not result.refused
    assert result.context_hits == []
    assert result.citations == []
    assert result.answer_text == "A general-knowledge answer."
    assert "No course materials were provided" in messages_seen[0][0]["content"]


def test_rag_answer_uses_prompt_context_and_refuses_unknown_citations(monkeypatch):
    chunks = [
        _chunk("chunk_a", "Exam weight is 40 percent.")
    ]
    monkeypatch.setattr(answer_module, "_build_hits", lambda q, cfg, filters: _mock_hits(chunks, q, cfg, filters))
    messages_seen = []

    def fake_chat(messages, cfg, stream=False):
        messages_seen.append(messages)
        return "It is 40 percent [S1], according to [S99]."

    monkeypatch.setattr(
        answer_module,
        "chat",
        fake_chat,
    )

    result = answer_module.answer("What is the exam weight?", cfg=_config())

    assert "[S1]" in messages_seen[0][-1]["content"]
    assert "Exam weight is 40 percent." in messages_seen[0][-1]["content"]
    assert result.refused
    assert result.answer_text == "This isn't covered in the course material I have."
    assert result.citations == []


def test_rag_answer_without_a_valid_citation_is_refused(monkeypatch):
    chunks = [
        _chunk("chunk_a", "Exam weight is 40 percent.")
    ]
    monkeypatch.setattr(answer_module, "_build_hits", lambda q, cfg, filters: _mock_hits(chunks, q, cfg, filters))
    monkeypatch.setattr(answer_module, "chat", lambda messages, cfg: "It is 40 percent.")

    result = answer_module.answer("What is the exam weight?", cfg=_config())

    assert result.refused
    assert result.answer_text == "This isn't covered in the course material I have."
    assert result.citations == []


def test_stream_preserves_answer_text_and_finishes_with_result(monkeypatch):
    chunks = [
        _chunk("chunk_a", "Exam weight is 40 percent.")
    ]
    monkeypatch.setattr(answer_module, "_build_hits", lambda q, cfg, filters: _mock_hits(chunks, q, cfg, filters))

    pieces = list(answer_module.answer_stream("Exam weight", cfg=_config()))

    assert isinstance(pieces[-1], AnswerResult)
    assert "".join(pieces[:-1]) == pieces[-1].answer_text


def test_out_of_scope_route_skips_retrieval(monkeypatch):
    monkeypatch.setattr(
        answer_module,
        "route_question",
        lambda question, cfg: RouteDecision(label="out_of_scope", prob=0.95, source="classifier"),
    )
    monkeypatch.setattr(
        answer_module,
        "_build_hits",
        lambda *args: pytest.fail("out-of-scope questions must skip retrieval"),
    )

    result = answer_module.answer("Question outside the materials", cfg=_config())

    assert result.refused
    assert result.route.label == "out_of_scope"
    assert result.timings_ms["route"] >= 0
    assert result.context_hits == []


def test_cross_course_route_uses_course_table(monkeypatch):
    hit = Hit(chunk=_chunk("course_row", "Management Control (c01)"), rank=1, label="S1")
    monkeypatch.setattr(
        answer_module,
        "route_question",
        lambda question, cfg: RouteDecision(label="cross_course", prob=0.9, source="classifier"),
    )
    monkeypatch.setattr(answer_module.table_qa, "answer_from_table", lambda question, cfg: [hit])
    monkeypatch.setattr(
        answer_module,
        "_build_hits",
        lambda *args: pytest.fail("table evidence should avoid normal retrieval"),
    )
    monkeypatch.setattr(
        answer_module,
        "build_messages",
        lambda question, hits, cfg, history: [{"role": "user", "content": hits[0].chunk.text}],
    )
    monkeypatch.setattr(answer_module, "chat", lambda messages, cfg: "Answer [S1]")

    result = answer_module.answer("Compare courses", cfg=_config())

    assert result.route.label == "cross_course"
    assert result.context_hits[0].chunk.chunk_id == hit.chunk.chunk_id
    assert result.citations[0].label == "S1"


def test_cross_course_route_falls_back_when_course_table_is_empty(monkeypatch):
    hit = Hit(chunk=_chunk("retrieved", "Exam weight is 40 percent."), rank=1)
    monkeypatch.setattr(
        answer_module,
        "route_question",
        lambda question, cfg: RouteDecision(label="cross_course", prob=0.9, source="classifier"),
    )
    monkeypatch.setattr(answer_module.table_qa, "answer_from_table", lambda question, cfg: [])
    monkeypatch.setattr(answer_module, "_build_hits", lambda question, cfg, filters: [hit])

    result = answer_module.answer("Compare courses", cfg=_config())

    assert result.context_hits[0].chunk.chunk_id == "retrieved"


def test_management_control_exam_evidence_is_passed_to_the_model(monkeypatch):
    from src.config import chunks_path, load_config
    chromadb = pytest.importorskip("chromadb")
    pytest.importorskip("sentence_transformers")
    from src.config import PATHS, bm25_path

    cfg = load_config("configs/final.yaml")
    if not chunks_path(cfg.kb_sig).is_file():
        pytest.skip("the generated chunk index is not available")
    if not bm25_path(cfg.kb_sig).is_file():
        pytest.skip("the configured BM25 index is not built")
    client = chromadb.PersistentClient(path=str(PATHS["chroma"]))
    if cfg.collection_name not in {
        item.name if hasattr(item, "name") else str(item)
        for item in client.list_collections()
    }:
        pytest.skip("the configured Chroma collection is not built")

    monkeypatch.setattr(
        answer_module,
        "chat",
        lambda messages, cfg: "The final exam is worth 40% [S1].",
    )
    result = answer_module.answer(
        "What is the final exam worth?",
        filters=Filters(course_code="mgmt_control"),
        cfg=cfg,
    )

    assert "40%" in result.answer_text
    assert result.citations
    assert any("40%" in hit.chunk.text for hit in result.context_hits)
