"""Question-answering API used by the app and evaluation.

Owner: Sergio.

Retrieves evidence from ChromaDB and BM25, then generates a cited answer locally.
"""
from __future__ import annotations

import copy
import re
import time
from dataclasses import replace
from typing import Iterator

from src.config import Config, load_config
from src.generation import rewrite, selfcheck, table_qa
from src.generation.llm import chat
from src.generation.prompts import build_messages
from src.retrieval import bm25, dense, fusion, rerank
from src.retrieval.filters import mentioned_course_codes
from src.retrieval.router import route as route_question
from src.schemas import AnswerResult, Citation, Filters, Hit, REFUSAL, RouteDecision

_CITATION = re.compile(r"[\[(]S(\d+)[\])]", re.IGNORECASE)
_COMPLETE_CITED_SENTENCE = re.compile(
    r"(?:\s*\[S\d+\])*[.!?](?:\s*\[S\d+\])*(?=\s|$)",
    re.IGNORECASE,
)
_COMPLETE_SENTENCE = re.compile(r"[.!?](?=\s|$)")


def _trim_to_last_complete_sentence(answer_text: str, rag_enabled: bool) -> str:
    """Drop a length-truncated tail that does not end in a complete sentence."""
    sentence_pattern = _COMPLETE_CITED_SENTENCE if rag_enabled else _COMPLETE_SENTENCE
    completed = list(sentence_pattern.finditer(answer_text))
    if not completed:
        return ""
    return answer_text[:completed[-1].end()].rstrip()


def _top_k(cfg: Config) -> int:
    top_k = cfg.retrieval.get("top_k", 5)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError(f"retrieval.top_k must be a positive integer; got {top_k!r}")
    return top_k


def _build_hits(question: str, cfg: Config, filters: Filters | None) -> list[Hit]:
    top_k = _top_k(cfg)
    mode = cfg.retrieval.get("mode", "hybrid")
    if mode not in {"dense", "hybrid"}:
        raise ValueError(f"unsupported retrieval.mode: {mode!r}")
    dense_k = cfg.retrieval.get("k_dense", top_k)
    if not isinstance(dense_k, int) or isinstance(dense_k, bool) or dense_k < 1:
        raise ValueError(f"retrieval.k_dense must be a positive integer; got {dense_k!r}")

    dense_hits = dense.search(question, cfg, filters, dense_k)
    if mode == "dense":
        return dense_hits

    bm25_k = cfg.retrieval.get("k_bm25", top_k)
    if not isinstance(bm25_k, int) or isinstance(bm25_k, bool) or bm25_k < 1:
        raise ValueError(f"retrieval.k_bm25 must be a positive integer; got {bm25_k!r}")
    bm25_hits = bm25.search(question, cfg, filters, bm25_k)
    rrf_k = cfg.retrieval.get("rrf_k", 60)
    if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k < 1:
        raise ValueError(f"retrieval.rrf_k must be a positive integer; got {rrf_k!r}")
    return fusion.rrf([dense_hits, bm25_hits], k=rrf_k)


def _course_search_hits(question: str, cfg: Config, table_rows: list[Hit]) -> list[Hit]:
    """1. Search results added to the course table rows for cross-course questions the table cannot answer.

    2. The course table holds weights, attendance, professor and sessions, but not midterm sessions,
       textbooks or case details, so "In which session is the midterm in Macroeconomics, Management
       Control and Time Series?" was refused with only the table rows as sources.
    3. Each course named in the rows gets its best search results (1 or 2), filtered to that course;
       with all 7 rows (no course named) the 3 best unfiltered results are added instead.
    """
    codes = list(dict.fromkeys(hit.chunk.course_code for hit in table_rows if hit.chunk.course_code))
    use_rerank = cfg.retrieval.get("rerank", False)

    def best(filters: Filters | None, keep: int) -> list[Hit]:
        # 4. only the 10 best fused candidates go to the reranker, to keep the extra time low
        candidates = _build_hits(question, cfg, filters)[:10]
        if not candidates:
            return []
        return rerank.rerank(question, candidates, keep, cfg) if use_rerank else candidates[:keep]

    if not codes or len(codes) >= 7:
        return best(None, 3)
    keep = 2 if len(codes) <= 2 else 1
    found = []
    for code in codes:
        found.extend(best(Filters(course_code=code), keep))
    return found


def _prepare_answer(
    question: str,
    history: list[dict] | None,
    filters: Filters | None,
    cfg: Config | None,
) -> tuple[
    str, str, Config, list[Hit], list[dict[str, str]] | None, RouteDecision,
    float, float, float, float, float, bool, Filters | None,
]:
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")

    cfg = cfg or load_config("configs/final_system.yaml")
    started = time.perf_counter()
    rewritten_question = question
    rag_enabled = cfg.generation.get("rag", True)
    rewrite_ms = 0.0
    translation_enabled = rag_enabled and cfg.generation.get("translate_query", False)
    if (history and cfg.generation.get("rewrite", False)) or translation_enabled:
        rewrite_started = time.perf_counter()
        translate_to_english = translation_enabled and rewrite.detect_language(question) != "en"
        if history and cfg.generation.get("rewrite", False):
            if translate_to_english:
                rewritten_question = rewrite.rewrite(
                    question,
                    history,
                    cfg,
                    translate_to_english=True,
                )
            else:
                rewritten_question = rewrite.rewrite(question, history, cfg)
        elif translate_to_english:
            rewritten_question = rewrite.translate_to_english(question, cfg)
        rewrite_ms = (time.perf_counter() - rewrite_started) * 1000

    route_ms = 0.0
    route_decision = RouteDecision(label="none", prob=0.0, source="none")
    if rag_enabled:
        route_started = time.perf_counter()
        route_decision = route_question(rewritten_question, cfg)
        route_ms = (time.perf_counter() - route_started) * 1000

    # 6. With course_split, a comparison of two or more named courses on a field the table holds goes to the
    #    course table even if the router said course_admin (SA10 was searched and its two weights mixed up).
    if (
        rag_enabled
        and cfg.generation.get("course_split", False)
        and route_decision.label not in ("cross_course", "out_of_scope")
        and len(_named_courses(rewritten_question)) >= 2
        and table_qa.asks_for_ranking(rewritten_question)
    ):
        route_decision = RouteDecision(label="cross_course", prob=1.0, source="course_split")

    # 8. With routing.named_course_in_scope, a question that names a course of the programme is never sent out
    #    of scope by the router: "In the beer market example in Strategy, how many surveys..." got 0.60 for
    #    out_of_scope and was refused without a search. It is searched, and the self-check and the model still
    #    refuse it when the material does not have the answer.
    if (
        rag_enabled
        and route_decision.label == "out_of_scope"
        and getattr(cfg, "routing", {}).get("named_course_in_scope", False)
        and _named_courses(rewritten_question)
    ):
        route_decision = RouteDecision(label="course_content", prob=route_decision.prob, source="named_course")

    # 7. With routing.out_of_scope_search, an "out of scope" label no longer refuses without searching: the
    #    question is searched and the self-check refuses it only if no source scores well. The router sent two
    #    answerable questions there ("Who teaches Operating Systems ... where did he work before IE?" and a
    #    question about the Linux directory tree in the slides), and nothing could correct it.
    if (
        rag_enabled
        and route_decision.label == "out_of_scope"
        and not getattr(cfg, "routing", {}).get("out_of_scope_search", False)
    ):
        return (
            question, rewritten_question, cfg, [], None, route_decision, started,
            rewrite_ms, route_ms, 0.0, 0.0, rag_enabled, filters,
        )

    # 1. With retrieval.auto_course_filter, a question that names exactly one course searches only that course,
    #    so slides of other courses (Macroeconomics and OB slides in a Strategies question) never reach the model.
    if (
        rag_enabled
        and cfg.retrieval.get("auto_course_filter", False)
        and not (filters and filters.course_code)
        and route_decision.label != "cross_course"
    ):
        named = _named_courses(rewritten_question)
        if len(named) == 1:
            filters = Filters(course_code=named[0], sessions=list(filters.sessions) if filters else [])

    retrieval_started = time.perf_counter()
    table_hits = False
    hits = []
    if rag_enabled:
        # 2. a cross-course question already split into one course (course_split) is answered from that course's search
        if route_decision.label == "cross_course" and not (filters and filters.course_code):
            hits = table_qa.answer_from_table(rewritten_question, cfg)
            table_hits = bool(hits)
            if table_hits and hits[0].chunk.chunk_id != "course_ranking":
                hits = hits + _course_search_hits(rewritten_question, cfg, hits)
        if not hits:
            hits = _build_hits(rewritten_question, cfg, filters)
    retrieve_ms = (time.perf_counter() - retrieval_started) * 1000
    rerank_ms = 0.0
    if rag_enabled and hits and not table_hits:
        top_k = _top_k(cfg)
        if cfg.retrieval.get("rerank", False):
            rerank_started = time.perf_counter()
            hits = rerank.rerank(rewritten_question, hits, top_k, cfg)
            rerank_ms = (time.perf_counter() - rerank_started) * 1000
        else:
            hits = hits[:top_k]
        is_weak = cfg.generation.get("selfcheck", False) and selfcheck.is_weak(hits, cfg)
        hits = [replace(hit, rank=rank, label=f"S{rank}") for rank, hit in enumerate(hits, start=1)]
    elif table_hits:
        is_weak = False
        hits = [replace(hit, rank=rank, label=f"S{rank}") for rank, hit in enumerate(hits, start=1)]
    else:
        is_weak = False
    messages = (
        build_messages(question, hits, cfg, history)
        if not rag_enabled or (hits and not is_weak)
        else None
    )
    return (
        question, rewritten_question, cfg, hits, messages, route_decision, started,
        rewrite_ms, route_ms, retrieve_ms, rerank_ms, rag_enabled, filters,
    )


def _answer_part(answer_text: str, cfg: Config) -> str:
    """8. With prompt v4 the model writes "Evidence: ... Answer: ..."; the student only sees the answer part.

    If the model forgets the "Answer:" line, the whole text is kept, so a correct answer is never lost.
    """
    if cfg.generation.get("prompt_version") != "v4":
        return answer_text
    match = re.search(r"answer\s*:\s*", answer_text, flags=re.IGNORECASE)
    return answer_text[match.end():].strip() if match else answer_text


def _ranking_answer(hits: list[Hit]) -> str | None:
    """1. Comparison questions are answered with the ranking line itself, without calling the model.

    2. Even with the ranking written out as S1, the 3B model kept merging the first two groups
       ("Strategies and Management Control, both 40%"; Strategies is 45%). The table already holds
       the right answer, so copying it is exact and also saves the generation time.
    """
    if hits and hits[0].chunk.chunk_id == "course_ranking":
        return f"{hits[0].chunk.text} [{hits[0].label}]"
    return None


_CITED_PART = re.compile(r"(?P<body>[^\[]*?)(?P<cites>(?:\s*\[S\d+\])+)(?P<end>\s*[.!?]?)")


def _merge_repeated_sentences(answer_text: str) -> str:
    """Write a sentence once when the model repeats it to cite each source that says the same thing.

    When several sources agree (the course summary and two slides all give "Final Exam: 40%"), the model
    sometimes writes the same sentence once per source: "X [S1]. X [S2]. X [S3]." This keeps the first
    one with all its citations, "X [S1][S2][S3].", so no fact and no citation is lost. Answers without an
    exact repeated sentence are returned unchanged.
    """
    parts = [m for m in _CITED_PART.finditer(answer_text) if m.group(0)]
    keys = [" ".join(m.group("body").lower().split()).strip(" .,;:!?") for m in parts]
    if len({k for k in keys if k}) == len([k for k in keys if k]):
        return answer_text  # nothing repeated
    merged: dict[str, list[str]] = {}  # sentence -> its labels, in order of appearance
    for match, key in zip(parts, keys):
        if key:
            labels = merged.setdefault(key, [])
            labels += [c for c in re.findall(r"\[S\d+\]", match.group("cites")) if c not in labels]
    pieces, written = [], set()
    for match, key in zip(parts, keys):
        if not key:
            pieces.append(match.group(0))
        elif key not in written:
            written.add(key)
            pieces.append(match.group("body").rstrip() + " " + "".join(merged[key]) + match.group("end"))
    tail = answer_text[parts[-1].end():] if parts else ""
    return ("".join(pieces) + tail).strip()


def _validated_answer(answer_text: str, hits: list[Hit], rag_enabled: bool) -> tuple[str, list[Hit]]:
    if not rag_enabled:
        return answer_text.strip(), []

    hits_by_label = {hit.label: hit for hit in hits}
    used_labels: list[str] = []
    invalid_citation = False

    def normalize_citation(match: re.Match[str]) -> str:
        nonlocal invalid_citation
        label = f"S{match.group(1)}"
        if label not in hits_by_label:
            invalid_citation = True
            return match.group(0)
        if label not in used_labels:
            used_labels.append(label)
        return f"[{label}]"

    answer_text = _CITATION.sub(normalize_citation, answer_text).strip()
    answer_text = _merge_repeated_sentences(answer_text)
    if invalid_citation or not used_labels or not answer_text:
        return REFUSAL, []
    return answer_text, [hits_by_label[label] for label in used_labels]


def _make_result(
    question: str,
    rewritten_question: str,
    route_decision: RouteDecision,
    filters: Filters | None,
    cfg: Config,
    hits: list[Hit],
    answer_text: str,
    started: float,
    rewrite_ms: float,
    route_ms: float,
    retrieve_ms: float,
    rerank_ms: float,
    generate_ms: float,
    rag_enabled: bool,
) -> AnswerResult:
    answer_text, cited_hits = _validated_answer(answer_text, hits, rag_enabled)
    citations = [
        Citation(label=hit.label, chunk_id=hit.chunk.chunk_id, covers=hit.chunk.covers)
        for hit in cited_hits
    ]
    return AnswerResult(
        question=question,
        rewritten_question=rewritten_question,
        route=route_decision,
        filters_used=filters or Filters(),
        context_hits=hits,
        answer_text=answer_text,
        citations=citations,
        refused=rag_enabled and answer_text == REFUSAL,
        timings_ms={
            "rewrite": round(rewrite_ms, 1),
            "route": round(route_ms, 1),
            "retrieve": round(retrieve_ms, 1),
            "rerank": round(rerank_ms, 1),
            "generate": round(generate_ms, 1),
            "total": round((time.perf_counter() - started) * 1000, 1),
        },
        config_name=cfg.name,
        kb_sig=cfg.kb_sig,
    )


# 5. Short names students use for two courses ("the Strategy course", "Machine Learning Foundations"), added only
#    for course_split and auto_course_filter so the evaluated matching of the frozen version stays as it was.
_COURSE_ALIASES = [
    {"course_code": "strategies", "course_name": "Strategy"},
    {"course_code": "ai_ml", "course_name": "Machine Learning Foundations"},
]


def _named_courses(question: str) -> list[str]:
    """Course codes named in the question, in order, each once (course table names plus the short names)."""
    return list(dict.fromkeys(mentioned_course_codes(question, table_qa.course_rows() + _COURSE_ALIASES)))


def _split_by_course(
    question: str,
    history: list[dict] | None,
    filters: Filters | None,
    cfg: Config,
) -> AnswerResult | None:
    """3. With generation.course_split, a question that names two or more courses is answered course by course.

    4. With the sources of several courses together, the model gave one course's midterm session for all three
       and called 35% and 40% "the same weight". Each course is now searched and answered on its own, with only
       its own sources, and the parts are joined with their labels renumbered (the second part's [S1] becomes [S6]).
       Ranking questions keep the exact ranking line from the course table.
    """
    if not cfg.generation.get("course_split", False) or not cfg.generation.get("rag", True):
        return None
    if filters and filters.course_code:
        return None
    rows = table_qa.course_rows()
    codes = _named_courses(question)
    if len(codes) < 2 or table_qa.asks_for_ranking(question):
        return None
    names = {row["course_code"]: row["course_name"] for row in rows}
    started = time.perf_counter()
    single = copy.copy(cfg)
    single.generation = {**cfg.generation, "course_split": False}
    parts, hits, timings = [], [], {}
    for code in codes:
        sub = answer(
            f"{question} Answer only for {names.get(code, code)}.",
            history,
            Filters(course_code=code, sessions=list(filters.sessions) if filters else []),
            single,
        )
        offset = len(hits)
        hits.extend(replace(hit, rank=offset + hit.rank, label=f"S{offset + hit.rank}") for hit in sub.context_hits)
        if sub.refused:
            parts.append(f"{names.get(code, code)}: the material does not cover it.")
        else:
            parts.append(re.sub(r"\[S(\d+)\]", lambda m: f"[S{int(m.group(1)) + offset}]", sub.answer_text))
        for step, ms in sub.timings_ms.items():
            timings[step] = timings.get(step, 0.0) + ms
    route = RouteDecision(label="cross_course", prob=1.0, source="course_split")
    return _make_result(
        question, question, route, filters, cfg, hits, " ".join(parts), started,
        timings.get("rewrite", 0.0), timings.get("route", 0.0), timings.get("retrieve", 0.0),
        timings.get("rerank", 0.0), timings.get("generate", 0.0), True,
    )


def answer(
    question: str,
    history: list[dict] | None = None,
    filters: Filters | None = None,
    cfg: Config | None = None,
) -> AnswerResult:
    """Generate an answer from retrieved evidence, or run the no-RAG baseline."""
    split = _split_by_course(question, history, filters, cfg or load_config("configs/final_system.yaml"))
    if split is not None:
        return split
    (
        question, rewritten_question, cfg, hits, messages, route_decision, started,
        rewrite_ms, route_ms, retrieve_ms, rerank_ms, rag_enabled, filters,
    ) = _prepare_answer(
        question, history, filters, cfg
    )
    if rag_enabled and messages is None:
        return _make_result(
            question, rewritten_question, route_decision, filters, cfg, hits, REFUSAL, started,
            rewrite_ms, route_ms, retrieve_ms, rerank_ms, 0.0, rag_enabled,
        )
    if messages is None:
        raise RuntimeError("answer prompt was not constructed")

    ranking_text = _ranking_answer(hits) if rag_enabled else None
    if ranking_text:
        return _make_result(
            question, rewritten_question, route_decision, filters, cfg, hits, ranking_text, started,
            rewrite_ms, route_ms, retrieve_ms, rerank_ms, 0.0, rag_enabled,
        )

    generation_started = time.perf_counter()
    response = chat(messages, cfg)
    if not isinstance(response, str):
        raise TypeError("chat() must return text when stream=False")
    answer_text = str(response)
    if getattr(response, "done_reason", None) == "length":
        answer_text = _trim_to_last_complete_sentence(answer_text, rag_enabled)
    if not answer_text.strip() and getattr(response, "done_reason", None) != "length":
        raise ValueError("chat() returned empty answer text")
    generate_ms = (time.perf_counter() - generation_started) * 1000
    answer_text = _answer_part(answer_text, cfg)
    return _make_result(
        question, rewritten_question, route_decision, filters, cfg, hits, answer_text, started,
        rewrite_ms, route_ms, retrieve_ms, rerank_ms, generate_ms, rag_enabled,
    )


def answer_stream(
    question: str,
    history: list[dict] | None = None,
    filters: Filters | None = None,
    cfg: Config | None = None,
) -> Iterator[str | AnswerResult]:
    """Yield generated text pieces and then the validated final result."""
    split = _split_by_course(question, history, filters, cfg or load_config("configs/final_system.yaml"))
    if split is not None:
        yield split.answer_text
        yield split
        return
    if (cfg or load_config("configs/final_system.yaml")).generation.get("prompt_version") == "v4":
        result = answer(question, history, filters, cfg)
        yield result.answer_text
        yield result
        return
    (
        question, rewritten_question, cfg, hits, messages, route_decision, started,
        rewrite_ms, route_ms, retrieve_ms, rerank_ms, rag_enabled, filters,
    ) = _prepare_answer(
        question, history, filters, cfg
    )
    if rag_enabled and messages is None:
        result = _make_result(
            question, rewritten_question, route_decision, filters, cfg, hits, REFUSAL, started,
            rewrite_ms, route_ms, retrieve_ms, rerank_ms, 0.0, rag_enabled,
        )
        yield result.answer_text
        yield result
        return
    if messages is None:
        raise RuntimeError("answer prompt was not constructed")

    ranking_text = _ranking_answer(hits) if rag_enabled else None
    if ranking_text:
        result = _make_result(
            question, rewritten_question, route_decision, filters, cfg, hits, ranking_text, started,
            rewrite_ms, route_ms, retrieve_ms, rerank_ms, 0.0, rag_enabled,
        )
        yield result.answer_text
        yield result
        return

    generation_started = time.perf_counter()
    response = chat(messages, cfg, stream=True)
    if isinstance(response, str):
        raise TypeError("chat() must return an iterator when stream=True")
    pieces: list[str] = []
    pending = ""
    for piece in response:
        pieces.append(piece)
        pending += piece
        sentence_pattern = _COMPLETE_CITED_SENTENCE if rag_enabled else _COMPLETE_SENTENCE
        completed = list(sentence_pattern.finditer(pending))
        if completed:
            end = completed[-1].end()
            yield pending[:end]
            pending = pending[end:]
    done_reason = getattr(response, "done_reason", None)
    if done_reason != "length" and pending:
        yield pending
    answer_text = "".join(pieces)
    if done_reason == "length":
        answer_text = _trim_to_last_complete_sentence(answer_text, rag_enabled)
    if not answer_text.strip() and done_reason != "length":
        raise ValueError("chat() returned no answer text while streaming")
    generate_ms = (time.perf_counter() - generation_started) * 1000
    result = _make_result(
        question, rewritten_question, route_decision, filters, cfg, hits, answer_text, started,
        rewrite_ms, route_ms, retrieve_ms, rerank_ms, generate_ms, rag_enabled,
    )
    if not answer_text.strip():
        yield result.answer_text
    yield result
