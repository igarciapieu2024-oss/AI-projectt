# Tests of the answer phase (src/generation/answer.py). The searches and the LLM are replaced by fakes, so the tests
# run without models or indexes.
import re
from types import SimpleNamespace

import pytest

import src.generation.answer as answer_module
from src.generation.llm import ChatText
from src.schemas import AnswerResult, Chunk, Filters, Hit, RouteDecision


# A small fake chunk for the tests.
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


# A small test configuration: hybrid search, no reranker and prompt v2.
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


# Fake search results, so no index is needed.
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


# Every test of this file uses a fake LLM that returns a fixed answer.
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


# The answer gets only top_k sources and cites the ones it used.
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


# Only the sources the model cited appear as citations.
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


# The course and session filters reach the search.
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


# When the search finds nothing, the answer is the refusal message.
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


# An empty question or a wrong top_k raises an error.
def test_invalid_question_and_top_k_are_rejected(monkeypatch):
    with pytest.raises(ValueError, match="must not be empty"):
        answer_module.answer("  ", cfg=_config())
    with pytest.raises(ValueError, match="positive integer"):
        answer_module.answer("Question", cfg=_config(top_k=0))


# With rag off, the model is called without searching.
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


# The sources go into the prompt, and an answer that cites a label that doesn't exist is refused.
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


# An answer without any valid citation is refused.
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


# An answer cut by the token limit keeps only its complete cited sentences.
def test_length_truncated_answer_keeps_only_last_complete_cited_sentence(monkeypatch):
    hit = _chunk("chunk_a", "Exam weight is 40 percent.")
    monkeypatch.setattr(
        answer_module,
        "_build_hits",
        lambda question, cfg, filters: [Hit(chunk=hit, rank=1)],
    )
    monkeypatch.setattr(
        answer_module,
        "chat",
        lambda messages, cfg: ChatText(
            "The exam is worth 40%. [S1] It is an important part of the grade [S1",
            "length",
        ),
    )

    result = answer_module.answer("Exam weight", cfg=_config())

    assert result.answer_text == "The exam is worth 40%. [S1]"
    assert [citation.label for citation in result.citations] == ["S1"]


# The same when the citation comes before the full stop.
def test_length_truncated_citation_before_period_keeps_completed_sentences(monkeypatch):
    hits = [
        Hit(chunk=_chunk("chunk_a", "Final exam is worth 40%."), rank=1),
        Hit(chunk=_chunk("chunk_b", "Attendance is 80%."), rank=2),
    ]
    monkeypatch.setattr(
        answer_module,
        "_build_hits",
        lambda question, cfg, filters: hits,
    )
    monkeypatch.setattr(
        answer_module,
        "chat",
        lambda messages, cfg: ChatText(
            "The final exam is worth 40% of the final grade [S1]. "
            "Attendance is 80% [S2]. Class participation will be",
            "length",
        ),
    )

    result = answer_module.answer("Course requirements", cfg=_config())

    assert result.answer_text == (
        "The final exam is worth 40% of the final grade [S1]. "
        "Attendance is 80% [S2]."
    )
    assert [citation.label for citation in result.citations] == ["S1", "S2"]


# A cut answer with no complete cited sentence is refused.
def test_length_truncated_answer_without_complete_citation_is_refused(monkeypatch):
    hit = _chunk("chunk_a", "Exam weight is 40 percent.")
    monkeypatch.setattr(
        answer_module,
        "_build_hits",
        lambda question, cfg, filters: [Hit(chunk=hit, rank=1)],
    )
    monkeypatch.setattr(
        answer_module,
        "chat",
        lambda messages, cfg: ChatText("The exam is worth 40% [S1", "length"),
    )

    result = answer_module.answer("Exam weight", cfg=_config())

    assert result.refused
    assert result.answer_text == "This isn't covered in the course material I have."


# Without RAG, a cut answer keeps its complete sentences.
def test_length_truncated_no_rag_answer_keeps_last_complete_sentence(monkeypatch):
    cfg = _config()
    cfg.generation = {"rag": False}
    monkeypatch.setattr(
        answer_module,
        "chat",
        lambda messages, config: ChatText(
            "A general answer is complete. Another thought is cut off",
            "length",
        ),
    )

    result = answer_module.answer("General question", cfg=cfg)

    assert result.answer_text == "A general answer is complete."


# The streaming version gives the same text and ends with the full result.
def test_stream_preserves_answer_text_and_finishes_with_result(monkeypatch):
    chunks = [
        _chunk("chunk_a", "Exam weight is 40 percent.")
    ]
    monkeypatch.setattr(answer_module, "_build_hits", lambda q, cfg, filters: _mock_hits(chunks, q, cfg, filters))

    pieces = list(answer_module.answer_stream("Exam weight", cfg=_config()))

    assert isinstance(pieces[-1], AnswerResult)
    assert "".join(pieces[:-1]) == pieces[-1].answer_text


# The streaming version never shows a sentence cut by the token limit.
def test_answer_stream_never_yields_incomplete_length_truncated_sentence(monkeypatch):
    hit = _chunk("chunk_a", "Exam weight is 40 percent.")
    monkeypatch.setattr(
        answer_module,
        "_build_hits",
        lambda question, cfg, filters: [Hit(chunk=hit, rank=1)],
    )

    class LimitedStream:
        done_reason = "length"

        def __iter__(self):
            return iter([
                "The exam is worth 40%. [S1] ",
                "This sentence is cut off [S1",
            ])

    monkeypatch.setattr(
        answer_module,
        "chat",
        lambda messages, cfg, stream=False: LimitedStream(),
    )

    pieces = list(answer_module.answer_stream("Exam weight", cfg=_config()))

    assert "".join(piece for piece in pieces[:-1] if isinstance(piece, str)) == (
        "The exam is worth 40%. [S1]"
    )
    assert isinstance(pieces[-1], AnswerResult)
    assert pieces[-1].answer_text == "The exam is worth 40%. [S1]"


# A sentence with its citation before the full stop is sent as soon as it's complete.
def test_answer_stream_emits_citation_before_period_sentences_promptly(monkeypatch):
    hits = [
        Hit(chunk=_chunk("chunk_a", "Final exam is worth 40%."), rank=1),
        Hit(chunk=_chunk("chunk_b", "Attendance is 80%."), rank=2),
    ]
    monkeypatch.setattr(
        answer_module,
        "_build_hits",
        lambda question, cfg, filters: hits,
    )

    class LimitedStream:
        done_reason = "length"

        def __init__(self):
            self.consumed = []

        def __iter__(self):
            for piece in [
                "The final exam is worth 40% of the final grade [S1]. ",
                "Attendance is 80% [S2]. ",
                "Class participation will be",
            ]:
                self.consumed.append(piece)
                yield piece

    response = LimitedStream()
    monkeypatch.setattr(
        answer_module,
        "chat",
        lambda messages, cfg, stream=False: response,
    )
    answer_pieces = answer_module.answer_stream("Course requirements", cfg=_config())

    first_piece = next(answer_pieces)

    assert first_piece == "The final exam is worth 40% of the final grade [S1]."
    assert len(response.consumed) == 1

    rest = list(answer_pieces)
    assert "".join([first_piece] + [
        piece for piece in rest[:-1] if isinstance(piece, str)
    ]) == (
        "The final exam is worth 40% of the final grade [S1]. "
        "Attendance is 80% [S2]."
    )
    assert isinstance(rest[-1], AnswerResult)
    assert rest[-1].answer_text == (
        "The final exam is worth 40% of the final grade [S1]. "
        "Attendance is 80% [S2]."
    )


# An out-of-scope question is refused without searching.
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


# A cross-course question takes its sources from the course table.
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
        lambda *args: [],
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



# A ranking question is answered with the ranking line, without calling the model.
def test_comparison_question_is_answered_with_the_ranking_line_without_the_model(monkeypatch):
    # 1. The 3B model merged the first two groups of the ranking, so the line is returned as it is
    ranking_text = "Highest final exam weight: Strategies (45%). Then: Management Control (40%)."
    ranking = Hit(chunk=_chunk("course_ranking", ranking_text, course_code=""), rank=1, label="S1")
    row = Hit(chunk=_chunk("course_row__c01", "Strategies (c01)"), rank=2, label="S2")
    monkeypatch.setattr(
        answer_module,
        "route_question",
        lambda question, cfg: RouteDecision(label="cross_course", prob=0.9, source="classifier"),
    )
    monkeypatch.setattr(answer_module.table_qa, "answer_from_table", lambda question, cfg: [ranking, row])
    monkeypatch.setattr(answer_module, "chat", lambda *args, **kwargs: pytest.fail("the model should not be called"))

    result = answer_module.answer("Which course has the highest final exam weight?", cfg=_config())
    assert result.answer_text == ranking_text + " [S1]"
    assert [citation.label for citation in result.citations] == ["S1"]
    assert not result.refused

    pieces = list(answer_module.answer_stream("Which course has the highest final exam weight?", cfg=_config()))
    assert pieces[0] == ranking_text + " [S1]"
    assert isinstance(pieces[-1], AnswerResult)

# Without a course table, a cross-course question is searched normally.
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


# The Management Control exam example: the evidence found reaches the model.
def test_management_control_exam_evidence_is_passed_to_the_model(monkeypatch):
    from src.config import chunks_path, load_config
    chromadb = pytest.importorskip("chromadb")
    pytest.importorskip("sentence_transformers")
    from src.config import PATHS, bm25_path

    cfg = load_config("configs/baselines/v1_2_small_model.yaml")
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


# Cross-course questions the table can't answer also get search results.
def test_cross_course_questions_the_table_cannot_answer_also_get_search_results(monkeypatch):
    # 1. the table rows come first, then the best search result of each named course
    rows = [
        Hit(chunk=_chunk("course_row__macro", "Macroeconomics: final exam none", course_code="macro"), rank=1),
        Hit(chunk=_chunk("course_row__ts", "Time Series: final exam 30%", course_code="ts"), rank=2),
    ]
    found = {
        "macro": _chunk("macro_syl_sec08", "Macroeconomics midterm exam in session 16", course_code="macro"),
        "ts": _chunk("ts_syl_sec08", "Time Series midterm exam in session 9", course_code="ts"),
    }
    monkeypatch.setattr(
        answer_module,
        "route_question",
        lambda question, cfg: RouteDecision(label="cross_course", prob=0.9, source="classifier"),
    )
    monkeypatch.setattr(answer_module.table_qa, "answer_from_table", lambda question, cfg: list(rows))
    monkeypatch.setattr(
        answer_module,
        "_build_hits",
        lambda question, cfg, filters: [Hit(chunk=found[filters.course_code], rank=1)] if filters else [],
    )
    seen = {}

    def fake_chat(messages, cfg, stream=False, **kwargs):
        seen["prompt"] = messages[-1]["content"]
        return "Macroeconomics: session 16 [S3]; Time Series: session 9 [S4]."

    monkeypatch.setattr(answer_module, "chat", fake_chat)
    result = answer_module.answer("In which session is the midterm in Macroeconomics and Time Series?", cfg=_config())

    assert not result.refused
    assert "session 16" in seen["prompt"] and "session 9" in seen["prompt"]
    assert [citation.label for citation in result.citations] == ["S3", "S4"]


# A configuration with course_split switched on.
def _split_config(**generation):
    cfg = _config()
    cfg.generation = {**cfg.generation, **generation}
    return cfg


# With course_split, each named course is answered with its own sources.
def test_course_split_answers_each_named_course_with_its_own_sources(monkeypatch):
    # 1. two courses named, not a ranking: one search per course, and the second part's labels are renumbered
    monkeypatch.setattr(answer_module.table_qa, "course_rows", lambda: [
        {"course_code": "macro", "course_name": "Macroeconomics"},
        {"course_code": "ts", "course_name": "Time Series"},
    ])
    monkeypatch.setattr(
        answer_module,
        "route_question",
        lambda question, cfg: RouteDecision(label="course_admin", prob=0.9, source="classifier"),
    )
    found = {
        "macro": _chunk("macro_s01_p017", "Macroeconomics midterm: session 16", course_code="macro"),
        "ts": _chunk("ts_syl_sec07", "Time Series midterm: session 9", course_code="ts"),
    }
    monkeypatch.setattr(
        answer_module,
        "_build_hits",
        lambda question, cfg, filters: [Hit(chunk=found[filters.course_code], rank=1)],
    )
    seen = []

    def fake_chat(messages, cfg, stream=False, **kwargs):
        content = messages[-1]["content"]
        seen.append(content)
        return "Session 16 [S1]." if "session 16" in content else "Session 9 [S1]."

    monkeypatch.setattr(answer_module, "chat", fake_chat)
    question = "In which session is the midterm in Macroeconomics and Time Series?"
    result = answer_module.answer(question, cfg=_split_config(course_split=True))

    assert len(seen) == 2 and all(("session 16" in c) != ("session 9" in c) for c in seen)
    assert result.answer_text == "Session 16 [S1]. Session 9 [S2]."
    assert [c.label for c in result.citations] == ["S1", "S2"]
    assert not result.refused


# course_split is off unless the config switches it on.
def test_course_split_is_off_by_default(monkeypatch):
    monkeypatch.setattr(answer_module.table_qa, "course_rows", lambda: [
        {"course_code": "macro", "course_name": "Macroeconomics"},
        {"course_code": "ts", "course_name": "Time Series"},
    ])
    assert answer_module._split_by_course("Midterm in Macroeconomics and Time Series?", None, None, _config()) is None


# With auto_course_filter, a question that names one course only searches that course.
def test_auto_course_filter_searches_only_the_one_course_named(monkeypatch):
    # 1. one course named and the option on: the search gets that course as filter
    monkeypatch.setattr(answer_module.table_qa, "course_rows", lambda: [
        {"course_code": "strategies", "course_name": "Strategies"},
        {"course_code": "macro", "course_name": "Macroeconomics"},
    ])
    monkeypatch.setattr(
        answer_module,
        "route_question",
        lambda question, cfg: RouteDecision(label="course_admin", prob=0.9, source="classifier"),
    )
    used = {}

    def fake_hits(question, cfg, filters):
        used["filters"] = filters
        return [Hit(chunk=_chunk("strategies_syl_sec08", "Quizzes: 25%", course_code="strategies"), rank=1)]

    monkeypatch.setattr(answer_module, "_build_hits", fake_hits)
    monkeypatch.setattr(answer_module, "chat", lambda *args, **kwargs: "25% [S1].")
    cfg = _config()
    cfg.retrieval = {**cfg.retrieval, "auto_course_filter": True}
    result = answer_module.answer("What percentage of the Strategy grade comes from quizzes?", cfg=cfg)
    assert used["filters"].course_code == "strategies"
    assert result.filters_used.course_code == "strategies"


# With out_of_scope_search, an out-of-scope question is searched and the self-check decides.
def test_out_of_scope_search_lets_the_self_check_decide(monkeypatch):
    # 1. with the option on, an "out of scope" label still searches; a strong source gets an answer
    monkeypatch.setattr(
        answer_module,
        "route_question",
        lambda question, cfg: RouteDecision(label="out_of_scope", prob=0.79, source="classifier"),
    )
    searched = []

    def fake_hits(question, cfg, filters):
        searched.append(question)
        return [Hit(chunk=_chunk("os_syl_sec02", "Professor: Matteo Turilli", course_code="os"), rank=1)]

    monkeypatch.setattr(answer_module, "_build_hits", fake_hits)
    monkeypatch.setattr(answer_module, "chat", lambda *args, **kwargs: "Matteo Turilli [S1].")
    cfg = _config()
    cfg.routing = {"out_of_scope_search": True}
    result = answer_module.answer("Who teaches Operating Systems?", cfg=cfg)
    assert searched and not result.refused


# With prompt v4, only the text after "Answer:" is shown.
def test_prompt_v4_shows_only_the_answer_part(monkeypatch):
    # 1. the model writes the copied evidence first; only what follows "Answer:" is kept and validated
    monkeypatch.setattr(
        answer_module,
        "route_question",
        lambda question, cfg: RouteDecision(label="course_admin", prob=0.9, source="classifier"),
    )
    monkeypatch.setattr(
        answer_module,
        "_build_hits",
        lambda question, cfg, filters: [Hit(chunk=_chunk("ob_s01_p019", "minimal passing grade of 4", course_code="ob"), rank=1)],
    )
    monkeypatch.setattr(
        answer_module,
        "chat",
        lambda *args, **kwargs: 'Evidence: "there is a minimal passing grade of 4" [S1]\nAnswer: Yes, at least 4 out of 10 [S1].',
    )
    cfg = _config()
    cfg.generation = {**cfg.generation, "prompt_version": "v4"}
    result = answer_module.answer("Is there a minimum final exam grade?", cfg=cfg)
    assert result.answer_text == "Yes, at least 4 out of 10 [S1]."
    assert [c.label for c in result.citations] == ["S1"]


# With named_course_in_scope, a question that names a course is never sent out of scope.
def test_named_course_is_never_out_of_scope(monkeypatch):
    # 1. the router said out_of_scope, but the question names Strategy, so it is searched and answered
    monkeypatch.setattr(
        answer_module,
        "route_question",
        lambda question, cfg: RouteDecision(label="out_of_scope", prob=0.6, source="classifier"),
    )
    searched = []

    def fake_hits(question, cfg, filters):
        searched.append(question)
        return [Hit(chunk=_chunk("strategies_s12_p053", "2,000 digital surveys. Source: DataCentric", course_code="strategies"), rank=1)]

    monkeypatch.setattr(answer_module, "_build_hits", fake_hits)
    monkeypatch.setattr(answer_module, "chat", lambda *args, **kwargs: "On 2,000 surveys; source DataCentric [S1].")
    cfg = _config()
    cfg.routing = {"named_course_in_scope": True}
    result = answer_module.answer("In the beer market example in Strategy, how many surveys is the map based on?", cfg=cfg)
    assert searched and not result.refused


# That rule doesn't change questions that name no course: they are still refused.
def test_named_course_rule_keeps_refusing_questions_without_a_course(monkeypatch):
    # 2. no course named: the out_of_scope label still refuses without a search
    monkeypatch.setattr(
        answer_module,
        "route_question",
        lambda question, cfg: RouteDecision(label="out_of_scope", prob=0.9, source="classifier"),
    )
    monkeypatch.setattr(answer_module, "_build_hits", lambda *a, **k: pytest.fail("must not search"))
    cfg = _config()
    cfg.routing = {"named_course_in_scope": True}
    result = answer_module.answer("What is the weather in Madrid tomorrow?", cfg=cfg)
    assert result.refused


def test_repeated_sentence_is_written_once_with_all_its_citations():
    """When several sources say the same thing, the model may repeat the sentence once per source."""
    from src.generation.answer import _merge_repeated_sentences

    repeated = (
        "The final exam is worth 40% of the final grade [S1]. "
        "The final exam is worth 40% of the final grade [S2]. "
        "The final exam is worth 40% of the final grade [S3]."
    )
    assert _merge_repeated_sentences(repeated) == "The final exam is worth 40% of the final grade [S1][S2][S3]."

    different = "The final exam is worth 40% [S1]. A minimum grade of 4.5 is required [S2]."
    assert _merge_repeated_sentences(different) == different
