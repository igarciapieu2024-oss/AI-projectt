# The answer phase: from a question to a cited answer. answer() is used by the evaluation (eval/run_eval.py)
# and answer_stream() by the app (app_components/answer_view.py); both run the same steps:
#   1. prepare the question (rewrite.py)          2. decide its path (router.py)
#   3. choose where to search (course filter, per-course answers, course table)
#   4. search ChromaDB and BM25 and merge them    5. rerank and run the self-check
#   6. build the prompt (prompts.py)              7. ask the LLM (llm.py) and check the citations
# Everything before the LLM is in _prepare_answer(), shared by both functions. The settings come from the
# config (configs/final_system.yaml by default).
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

# A citation written by the model: S and a number in square brackets or parentheses, such as [S1] or (s2).
_CITATION = re.compile(r"[\[(]S(\d+)[\])]", re.IGNORECASE)
# The end of a sentence, with the citations that may come before or after its full stop.
_COMPLETE_CITED_SENTENCE = re.compile(
    r"(?:\s*\[S\d+\])*[.!?](?:\s*\[S\d+\])*(?=\s|$)",
    re.IGNORECASE,
)
# The end of a sentence without citations (used for answers without RAG).
_COMPLETE_SENTENCE = re.compile(r"[.!?](?=\s|$)")


# When the answer was cut by the token limit (done_reason "length"), drop the unfinished last sentence,
# so the student never sees half a sentence. Returns "" if no sentence was completed.
def _trim_to_last_complete_sentence(answer_text: str, rag_enabled: bool) -> str:
    sentence_pattern = _COMPLETE_CITED_SENTENCE if rag_enabled else _COMPLETE_SENTENCE
    completed = list(sentence_pattern.finditer(answer_text))
    if not completed:
        return ""
    return answer_text[:completed[-1].end()].rstrip()


# The number of sources given to the LLM (retrieval.top_k, 5 in base.yaml).
def _top_k(cfg: Config) -> int:
    top_k = cfg.retrieval.get("top_k", 5)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError(f"retrieval.top_k must be a positive integer; got {top_k!r}")
    return top_k


# Search the knowledge base: dense search (ChromaDB) and BM25, merged with RRF (fusion.py). With
# retrieval.mode "dense" only the dense search is used (the first version of the system). Returns the
# merged candidates, which the reranker reduces to the best 5.
def _build_hits(question: str, cfg: Config, filters: Filters | None) -> list[Hit]:
    top_k = _top_k(cfg)
    mode = cfg.retrieval.get("mode", "hybrid")
    if mode not in {"dense", "hybrid"}:
        raise ValueError(f"unsupported retrieval.mode: {mode!r}")
    # How many candidates each search returns (k_dense and k_bm25, 20 in base.yaml).
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


# Search results added to the course table rows, for cross-course questions the table alone cannot answer.
# The table holds weights, attendance, professor and sessions, but not details such as the session of a
# midterm or the textbook, and such questions were refused with only the table rows as sources.
# Each course named in the rows gets its best 1 or 2 search results, filtered to that course; when all 7
# rows are there (no course named), the 3 best unfiltered results are added instead.
def _course_search_hits(question: str, cfg: Config, table_rows: list[Hit]) -> list[Hit]:
    codes = list(dict.fromkeys(hit.chunk.course_code for hit in table_rows if hit.chunk.course_code))
    use_rerank = cfg.retrieval.get("rerank", False)

    # The best `keep` results for one search; only the 10 best fused candidates go to the reranker, to keep
    # the extra time low.
    def best(filters: Filters | None, keep: int) -> list[Hit]:
        candidates = _build_hits(question, cfg, filters)[:10]
        if not candidates:
            return []
        return rerank.rerank(question, candidates, keep, cfg) if use_rerank else candidates[:keep]

    if not codes or len(codes) >= 7:
        return best(None, 3)
    # 2 results per course when there are one or two courses, 1 when there are more, so the prompt stays short.
    keep = 2 if len(codes) <= 2 else 1
    found = []
    for code in codes:
        found.extend(best(Filters(course_code=code), keep))
    return found


# Everything before the LLM: prepare the question, route it, choose the filters, search, rerank, run the
# self-check and build the prompt. Returns the original and rewritten question, the config, the sources (hits),
# the messages for the LLM (None when the answer must be a refusal), the route, the start time, the time of
# each step, whether RAG is on and the filters used. answer() and answer_stream() call it.
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
    # Step 1: prepare the question. The rewritten question is used for routing and searching; the LLM
    # still receives the original question and the history.
    rewritten_question = question
    rag_enabled = cfg.generation.get("rag", True)
    rewrite_ms = 0.0
    # Translation is only done when RAG is on and generation.translate_query is true.
    translation_enabled = rag_enabled and cfg.generation.get("translate_query", False)
    if (history and cfg.generation.get("rewrite", False)) or translation_enabled:
        rewrite_started = time.perf_counter()
        translate_to_english = translation_enabled and rewrite.detect_language(question) != "en"
        # A follow-up is rewritten with the history (and translated in the same call if needed); a new
        # question is only translated, and only if it is not in English.
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

    # Step 2: route the question (classifier, or the LLM when it is unsure). Without RAG there is no routing.
    route_ms = 0.0
    route_decision = RouteDecision(label="none", prob=0.0, source="none")
    if rag_enabled:
        route_started = time.perf_counter()
        route_decision = route_question(rewritten_question, cfg)
        route_ms = (time.perf_counter() - route_started) * 1000

    # Step 3: choose where to search.
    # With course_split, a question that compares two or more named courses on a field the table holds goes
    # to the course table even if the router said course_admin; searched normally, the weights of the two
    # courses were mixed up.
    if (
        rag_enabled
        and cfg.generation.get("course_split", False)
        and route_decision.label not in ("cross_course", "out_of_scope")
        and len(_named_courses(rewritten_question)) >= 2
        and table_qa.asks_for_ranking(rewritten_question)
    ):
        route_decision = RouteDecision(label="cross_course", prob=1.0, source="course_split")

    # With routing.named_course_in_scope (tested in experiments 09 and 10, not in the final system), a question that
    # names a course is never sent out of scope by the router: it is searched, and the self-check and the
    # model still refuse it when the material does not have the answer.
    if (
        rag_enabled
        and route_decision.label == "out_of_scope"
        and getattr(cfg, "routing", {}).get("named_course_in_scope", False)
        and _named_courses(rewritten_question)
    ):
        route_decision = RouteDecision(label="course_content", prob=route_decision.prob, source="named_course")

    # Out-of-scope questions are refused here, without searching. With routing.out_of_scope_search (tested in
    # experiments 06 to 08, not in the final system) they are searched instead and only refused by the self-check,
    # because the router had sent two answerable questions out of scope and nothing could correct it.
    if (
        rag_enabled
        and route_decision.label == "out_of_scope"
        and not getattr(cfg, "routing", {}).get("out_of_scope_search", False)
    ):
        return (
            question, rewritten_question, cfg, [], None, route_decision, started,
            rewrite_ms, route_ms, 0.0, 0.0, rag_enabled, filters,
        )

    # With retrieval.auto_course_filter (final_system.yaml), a question that names exactly one course searches
    # only that course, so slides of other courses never reach the model.
    if (
        rag_enabled
        and cfg.retrieval.get("auto_course_filter", False)
        and not (filters and filters.course_code)
        and route_decision.label != "cross_course"
    ):
        named = _named_courses(rewritten_question)
        if len(named) == 1:
            filters = Filters(course_code=named[0], sessions=list(filters.sessions) if filters else [])

    # Step 4: search. table_hits is True when the sources come from the course table.
    retrieval_started = time.perf_counter()
    table_hits = False
    hits = []
    if rag_enabled:
        # A cross-course question is answered from the course table, plus search results when the table rows
        # are not a ranking. A cross-course question already limited to one course (by course_split) is
        # searched normally instead.
        if route_decision.label == "cross_course" and not (filters and filters.course_code):
            hits = table_qa.answer_from_table(rewritten_question, cfg)
            table_hits = bool(hits)
            if table_hits and hits[0].chunk.chunk_id != "course_ranking":
                hits = hits + _course_search_hits(rewritten_question, cfg, hits)
        # Every other question, or a table with no rows: the normal search.
        if not hits:
            hits = _build_hits(rewritten_question, cfg, filters)
    retrieve_ms = (time.perf_counter() - retrieval_started) * 1000
    # Step 5: rerank the search results to the best top_k and run the self-check. Table rows are not reranked:
    # they are already exactly the rows the question needs.
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
        # Label the sources S1, S2... in their final order; these are the labels the model must cite.
        hits = [replace(hit, rank=rank, label=f"S{rank}") for rank, hit in enumerate(hits, start=1)]
    elif table_hits:
        is_weak = False
        hits = [replace(hit, rank=rank, label=f"S{rank}") for rank, hit in enumerate(hits, start=1)]
    else:
        is_weak = False
    # Step 6: build the prompt, unless there are no sources or they are too weak (then None = refusal).
    # Without RAG (baseline B0) the prompt is always built, without sources.
    messages = (
        build_messages(question, hits, cfg, history)
        if not rag_enabled or (hits and not is_weak)
        else None
    )
    return (
        question, rewritten_question, cfg, hits, messages, route_decision, started,
        rewrite_ms, route_ms, retrieve_ms, rerank_ms, rag_enabled, filters,
    )


# With prompt v4 (experiment 08) the model writes "Evidence: ... Answer: ...", and the student only sees the
# part after "Answer:". If the model forgets that line, the whole text is kept, so a correct answer is never
# lost. With the other prompt versions the text is returned unchanged.
def _answer_part(answer_text: str, cfg: Config) -> str:
    if cfg.generation.get("prompt_version") != "v4":
        return answer_text
    match = re.search(r"answer\s*:\s*", answer_text, flags=re.IGNORECASE)
    return answer_text[match.end():].strip() if match else answer_text


# Ranking questions are answered with the ranking line itself (built by table_qa.py), without calling the
# model. Even with the ranking written out as S1, the small model kept merging the first two groups; the
# table already holds the right answer, so copying it is exact and also saves the generation time.
# Returns None when the first source is not a ranking line.
def _ranking_answer(hits: list[Hit]) -> str | None:
    if hits and hits[0].chunk.chunk_id == "course_ranking":
        return f"{hits[0].chunk.text} [{hits[0].label}]"
    return None


# One sentence of the answer with the citations at its end and its final punctuation.
_CITED_PART = re.compile(r"(?P<body>[^\[]*?)(?P<cites>(?:\s*\[S\d+\])+)(?P<end>\s*[.!?]?)")


# Write a sentence once when the model repeats it to cite each source that says the same thing. When several
# sources agree, the model sometimes writes the same sentence once per source, each with a different citation.
# The first copy is kept with all the citations, so no fact and no citation is lost. Answers without an exact
# repeated sentence are returned unchanged. Only changes how the answer is shown, not what it says.
def _merge_repeated_sentences(answer_text: str) -> str:
    parts = [m for m in _CITED_PART.finditer(answer_text) if m.group(0)]
    # Each sentence compared in lower case, without extra spaces or final punctuation.
    keys = [" ".join(m.group("body").lower().split()).strip(" .,;:!?") for m in parts]
    if len({k for k in keys if k}) == len([k for k in keys if k]):
        return answer_text  # nothing repeated
    merged: dict[str, list[str]] = {}  # sentence -> its labels, in order of appearance
    for match, key in zip(parts, keys):
        if key:
            labels = merged.setdefault(key, [])
            labels += [c for c in re.findall(r"\[S\d+\]", match.group("cites")) if c not in labels]
    # Rebuild the answer: each sentence the first time it appears, with all its citations.
    pieces, written = [], set()
    for match, key in zip(parts, keys):
        if not key:
            pieces.append(match.group(0))
        elif key not in written:
            written.add(key)
            pieces.append(match.group("body").rstrip() + " " + "".join(merged[key]) + match.group("end"))
    # Text after the last cited sentence is kept as it is.
    tail = answer_text[parts[-1].end():] if parts else ""
    return ("".join(pieces) + tail).strip()


# Check the citations of the answer (the last part of step 7). Every [S..] must be one of the sources the
# model received. An answer with an invented label, or with no citation at all, is replaced by the refusal
# message, so the student never sees a claim that cannot be traced to a source. Returns the answer and the
# sources it cites. Without RAG there is nothing to check.
def _validated_answer(answer_text: str, hits: list[Hit], rag_enabled: bool) -> tuple[str, list[Hit]]:
    if not rag_enabled:
        return answer_text.strip(), []

    hits_by_label = {hit.label: hit for hit in hits}
    used_labels: list[str] = []
    invalid_citation = False

    # Write every citation as [S1] (also "(S1)" or "[s1]"), remember which labels were used and flag any
    # label that is not one of the sources.
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


# Build the AnswerResult (src/schemas.py) that the app shows and the evaluation saves: the answer after the
# citation check, the cited sources, all the sources, the route, the filters and the time of each step.
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
        # refused is True when the final answer is the refusal message (used by the evaluation metrics).
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


# Short names students use for two courses, added to the course names of the table only when finding the
# courses a question names (course filter and per-course answers), so the matching of the earlier evaluated
# versions stays as it was.
_COURSE_ALIASES = [
    {"course_code": "strategies", "course_name": "Strategy"},
    {"course_code": "ai_ml", "course_name": "Machine Learning Foundations"},
]


# The codes of the courses the question names, each once, using the course table names and the short names.
def _named_courses(question: str) -> list[str]:
    return list(dict.fromkeys(mentioned_course_codes(question, table_qa.course_rows() + _COURSE_ALIASES)))


# With generation.course_split (final_system.yaml), a question that names two or more courses is answered
# course by course. With the sources of several courses together, the model gave one course's fact for all of
# them. Each course is searched and answered on its own, with only its own sources, and the parts are joined
# with their labels renumbered, so every citation still points to the right source. Ranking questions are not
# split: they keep the ranking line of the course table. Returns None when the question is not split.
def _split_by_course(
    question: str,
    history: list[dict] | None,
    filters: Filters | None,
    cfg: Config,
) -> AnswerResult | None:
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
    # A copy of the config with course_split off, so each per-course answer is not split again.
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
        # Renumber this course's sources after the ones already collected, and its citations the same way.
        offset = len(hits)
        hits.extend(replace(hit, rank=offset + hit.rank, label=f"S{offset + hit.rank}") for hit in sub.context_hits)
        if sub.refused:
            parts.append(f"{names.get(code, code)}: the material does not cover it.")
        else:
            parts.append(re.sub(r"\[S(\d+)\]", lambda m: f"[S{int(m.group(1)) + offset}]", sub.answer_text))
        # The time of each step is the sum over the courses.
        for step, ms in sub.timings_ms.items():
            timings[step] = timings.get(step, 0.0) + ms
    route = RouteDecision(label="cross_course", prob=1.0, source="course_split")
    return _make_result(
        question, question, route, filters, cfg, hits, " ".join(parts), started,
        timings.get("rewrite", 0.0), timings.get("route", 0.0), timings.get("retrieve", 0.0),
        timings.get("rerank", 0.0), timings.get("generate", 0.0), True,
    )


# Answer a question and return the complete AnswerResult. Used by the evaluation (eval/run_eval.py).
# history is the conversation so far (for follow-ups), filters can force a course, and cfg is the
# configuration (final_system.yaml if none is given). Without RAG (baseline B0) the LLM answers alone.
def answer(
    question: str,
    history: list[dict] | None = None,
    filters: Filters | None = None,
    cfg: Config | None = None,
) -> AnswerResult:
    split = _split_by_course(question, history, filters, cfg or load_config("configs/final_system.yaml"))
    if split is not None:
        return split
    (
        question, rewritten_question, cfg, hits, messages, route_decision, started,
        rewrite_ms, route_ms, retrieve_ms, rerank_ms, rag_enabled, filters,
    ) = _prepare_answer(
        question, history, filters, cfg
    )
    # No prompt means no sources or weak sources: the answer is the refusal message, without calling the LLM.
    if rag_enabled and messages is None:
        return _make_result(
            question, rewritten_question, route_decision, filters, cfg, hits, REFUSAL, started,
            rewrite_ms, route_ms, retrieve_ms, rerank_ms, 0.0, rag_enabled,
        )
    if messages is None:
        raise RuntimeError("answer prompt was not constructed")

    # Ranking questions: the ranking line is the answer, without calling the LLM.
    ranking_text = _ranking_answer(hits) if rag_enabled else None
    if ranking_text:
        return _make_result(
            question, rewritten_question, route_decision, filters, cfg, hits, ranking_text, started,
            rewrite_ms, route_ms, retrieve_ms, rerank_ms, 0.0, rag_enabled,
        )

    # Step 7: ask the LLM.
    generation_started = time.perf_counter()
    response = chat(messages, cfg)
    if not isinstance(response, str):
        raise TypeError("chat() must return text when stream=False")
    answer_text = str(response)
    # An answer cut by the token limit loses its unfinished last sentence.
    if getattr(response, "done_reason", None) == "length":
        answer_text = _trim_to_last_complete_sentence(answer_text, rag_enabled)
    # An empty answer that was not cut means something went wrong in Ollama.
    if not answer_text.strip() and getattr(response, "done_reason", None) != "length":
        raise ValueError("chat() returned empty answer text")
    generate_ms = (time.perf_counter() - generation_started) * 1000
    answer_text = _answer_part(answer_text, cfg)
    # _make_result() checks the citations and builds the result.
    return _make_result(
        question, rewritten_question, route_decision, filters, cfg, hits, answer_text, started,
        rewrite_ms, route_ms, retrieve_ms, rerank_ms, generate_ms, rag_enabled,
    )


# The same steps as answer(), for the app: it yields the answer in pieces while the LLM writes it, and at the
# end the complete AnswerResult. A piece is only sent when a sentence (with its citations) is complete, so the
# student never sees a half-written citation.
def answer_stream(
    question: str,
    history: list[dict] | None = None,
    filters: Filters | None = None,
    cfg: Config | None = None,
) -> Iterator[str | AnswerResult]:
    # Per-course answers are not streamed: their parts are joined first, then sent at once.
    split = _split_by_course(question, history, filters, cfg or load_config("configs/final_system.yaml"))
    if split is not None:
        yield split.answer_text
        yield split
        return
    # With prompt v4 the evidence part must not be shown, so the answer is produced whole by answer().
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
    # Step 7 with streaming: the LLM sends its answer in small pieces.
    response = chat(messages, cfg, stream=True)
    if isinstance(response, str):
        raise TypeError("chat() must return an iterator when stream=True")
    # pieces: everything received; pending: text received but not shown yet, because its sentence is not complete.
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
    # The rest of the text, unless the answer was cut by the token limit (then the unfinished part is dropped).
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
    # If nothing was shown (for example, every sentence was cut), show the final text, then send the result.
    # The app replaces the streamed text with result.answer_text, which has passed the citation check.
    if not answer_text.strip():
        yield result.answer_text
    yield result
