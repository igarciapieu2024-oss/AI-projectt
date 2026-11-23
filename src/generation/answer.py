
# Retrieves evidence from ChromaDB and BM25, then generates a cited answer locally.

from __future__ import annotations

import copy                      # allows for creating true duplicates for mutable objects
import re                        # used for searching, matching, extracting, manipulating text based on pattern in the text in question
import time                      # for timing
from dataclasses import replace  #
from typing import Iterator

from src.config import Config, load_config
from src.generation import rewrite, selfcheck, table_qa
from src.generation.llm import chat
from src.generation.prompts import build_messages
from src.retrieval import bm25, dense, fusion, rerank
from src.retrieval.filters import mentioned_course_codes
from src.retrieval.router import route as route_question
from src.schemas import AnswerResult, Citation, Filters, Hit, REFUSAL, RouteDecision

# Rules for recognizing or rewriting text in the answer:

_CITATION = re.compile(r"[\[(]S(\d+)[\])]", re.IGNORECASE) # finds source labels

_COMPLETE_CITED_SENTENCE = re.compile(                     # finds objects before or after a punctuation
    r"(?:\s*\[S\d+\])*[.!?](?:\s*\[S\d+\])*(?=\s|$)",
    re.IGNORECASE,
)

_COMPLETE_SENTENCE = re.compile(r"[.!?](?=\s|$)")          # detects the end of standard sentences


# Removes the unfinished end of an answer (when the output limit cuts it off mid-sentence)

def _trim_to_last_complete_sentence(answer_text: str, rag_enabled: bool) -> str:
    """Drop a length-truncated tail that does not end in a complete sentence."""
    sentence_pattern = _COMPLETE_CITED_SENTENCE if rag_enabled else _COMPLETE_SENTENCE
    completed = list(sentence_pattern.finditer(answer_text))
    if not completed:
        return ""
    return answer_text[:completed[-1].end()].rstrip()


# Makes sure the value for top_k is a valid number

def _top_k(cfg: Config) -> int:
    top_k = cfg.retrieval.get("top_k", 5)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError(f"retrieval.top_k must be a positive integer; got {top_k!r}")
    return top_k


# Performing retrieval

def _build_hits(question: str, cfg: Config, filters: Filters | None) -> list[Hit]:

    top_k = _top_k(cfg)

    mode = cfg.retrieval.get("mode", "hybrid") # validates & saves the kind of search that will be performed
    if mode not in {"dense", "hybrid"}:
        raise ValueError(f"unsupported retrieval.mode: {mode!r}")

    dense_k = cfg.retrieval.get("k_dense", top_k) # validates & saves the number of candidates to retrieve for embedding search
    if not isinstance(dense_k, int) or isinstance(dense_k, bool) or dense_k < 1:
        raise ValueError(f"retrieval.k_dense must be a positive integer; got {dense_k!r}")

    dense_hits = dense.search(question, cfg, filters, dense_k) # performs the dense search (searches by comparing embeddings/vectors)
    if mode == "dense":
        return dense_hits

    bm25_k = cfg.retrieval.get("k_bm25", top_k) # validates & saves the number of candidates to retrieve for keyword search
    if not isinstance(bm25_k, int) or isinstance(bm25_k, bool) or bm25_k < 1:
        raise ValueError(f"retrieval.k_bm25 must be a positive integer; got {bm25_k!r}")

    bm25_hits = bm25.search(question, cfg, filters, bm25_k) # performs the keyword search (BM25, searches by keywords )

    rrf_k = cfg.retrieval.get("rrf_k", 60) # validates & saves the reciprocal-rank-fusion constant
    if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k < 1:
        raise ValueError(f"retrieval.rrf_k must be a positive integer; got {rrf_k!r}")

    return fusion.rrf([dense_hits, bm25_hits], k=rrf_k) # combines the dense and keyword search into one list


# Function adds course-material chunks when course-table answers (summaries of the course) are not enough to answer questions
# Searches the slides and syllabi for extra supporting chunks limited to the courses.

def _course_search_hits(question: str, cfg: Config, table_rows: list[Hit]) -> list[Hit]:

    # extracts the course codes from course-table (summaries) rows that are relevant to the question
    codes = list(dict.fromkeys(hit.chunk.course_code for hit in table_rows if hit.chunk.course_code))
    use_rerank = cfg.retrieval.get("rerank", False) # checks if reranking is enabled in the configurations

    # a function that can retrieve candidates again (this time with the filter of the course)
    def best(filters: Filters | None, keep: int) -> list[Hit]:
        candidates = _build_hits(question, cfg, filters)[:10] # performs retrieval and keeps 10 highest candidates
        if not candidates:
            return []
        return rerank.rerank(question, candidates, keep, cfg) if use_rerank else candidates[:keep]

    # decides how many extra chunks to retrieve per course
    if not codes or len(codes) >= 7: # if no course codes are available / all courses are involved
        return best(None, 3)         # uses the helper function to find 3 best chunks

    keep = 2 if len(codes) <= 2 else 1 # "keep" is the number of chunks per course (keeps 1 if there are more than 2 courses)

    found = [] # "found" will become a list of chunks with matching course codes to the question

    for code in codes:
        found.extend(best(Filters(course_code=code), keep)) # runs the "best" function and keeps a limited number per course
    return found                                            # gives the chunks that are found


# Setup and evidence-gathering stage
# Prepares everything for answering a question

def _prepare_answer(
    question: str,
    history: list[dict] | None,
    filters: Filters | None,
    cfg: Config | None,
) -> tuple[
    str, str, Config, list[Hit], list[dict[str, str]] | None, RouteDecision,
    float, float, float, float, float, bool, Filters | None,
]:
    question = question.strip() # cleans and initializes the question (removes spaces around the question)
    if not question:
        raise ValueError("question must not be empty")

    cfg = cfg or load_config("configs/final.yaml")
    started = time.perf_counter()
    rewritten_question = question
    rag_enabled = cfg.generation.get("rag", True)
    rewrite_ms = 0.0
    translation_enabled = rag_enabled and cfg.generation.get("translate_query", False)

    # translates the question (if config enables it)
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

    # ROUTING DECISION:

    route_ms = 0.0 # just for timing
    route_decision = RouteDecision(label="none", prob=0.0, source="none") # initially no route decision

    if rag_enabled:
        route_started = time.perf_counter()
        route_decision = route_question(rewritten_question, cfg) # classifies the rewritten question
        route_ms = (time.perf_counter() - route_started) * 1000
    #

    #
    if (
        rag_enabled
        and cfg.generation.get("course_split", False) # if course_split is enabled
        and route_decision.label not in ("cross_course", "out_of_scope")
        and len(_named_courses(rewritten_question)) >= 2 # at least two courses named
        and table_qa.asks_for_ranking(rewritten_question)
    ):
        # classifies it as cross_course
        route_decision = RouteDecision(label="cross_course", prob=1.0, source="course_split")
    #

    #
    if (
        rag_enabled
        and route_decision.label == "out_of_scope" # if out of scope
        and getattr(cfg, "routing", {}).get("named_course_in_scope", False) # if named_course_in_scope is enabled
        and _named_courses(rewritten_question) # if at least one known course is named
    ):
        # Overrides the router's out-of-scope label to course_content.
        route_decision = RouteDecision(label="course_content", prob=route_decision.prob, source="named_course")
    #

    # if the question is out_of_scope
    if (
        rag_enabled
        and route_decision.label == "out_of_scope" # if out of scope
        and not getattr(cfg, "routing", {}).get("out_of_scope_search", False)
    ):
        return (
            question, rewritten_question, cfg, [], None, route_decision, started,
            rewrite_ms, route_ms, 0.0, 0.0, rag_enabled, filters,
        ) # refusal
    #

    #
    if (
        rag_enabled
        and cfg.retrieval.get("auto_course_filter", False)
        and not (filters and filters.course_code)
        and route_decision.label != "cross_course"
    ):
        named = _named_courses(rewritten_question)
        if len(named) == 1:
            filters = Filters(course_code=named[0], sessions=list(filters.sessions) if filters else [])
    # limits the filter to one course if there is one named course in the rewritten question

    retrieval_started = time.perf_counter()
    table_hits = False
    hits = []

    #
    if rag_enabled:

        if route_decision.label == "cross_course" and not (filters and filters.course_code): # if no filter
            hits = table_qa.answer_from_table(rewritten_question, cfg)
            # looks up relevant course-table rows and returns them as hits
            table_hits = bool(hits) # records if any results were returned

            # if results were found and the first isn’t a prewritten ranking result
            if table_hits and hits[0].chunk.chunk_id != "course_ranking":
                hits = hits + _course_search_hits(rewritten_question, cfg, hits)
                # searches course materials for additional chunks

        if not hits:
            hits = _build_hits(rewritten_question, cfg, filters)
            # if hits is empty, it performs regular search (dense & BM25)
    #

    retrieve_ms = (time.perf_counter() - retrieval_started) * 1000
    rerank_ms = 0.0

    #
    if rag_enabled and hits and not table_hits: # if using regular retrieval results, not course-table results

        top_k = _top_k(cfg) # chunks to keep

        if cfg.retrieval.get("rerank", False): # if reranking is enabled
            rerank_started = time.perf_counter()
            hits = rerank.rerank(rewritten_question, hits, top_k, cfg) # rerank
            rerank_ms = (time.perf_counter() - rerank_started) * 1000
        else:
            hits = hits[:top_k] # if rerank is disabled (keep first top_k chunks in existing order)

                                                             # checks if the reranker score is below the threshold
        is_weak = cfg.generation.get("selfcheck", False) and selfcheck.is_weak(hits, cfg)
        hits = [replace(hit, rank=rank, label=f"S{rank}") for rank, hit in enumerate(hits, start=1)] # keep chunks in position

    elif table_hits: # if course_table lookup found results
        is_weak = False
        hits = [replace(hit, rank=rank, label=f"S{rank}") for rank, hit in enumerate(hits, start=1)] # keep chunks in position

    else:
        is_weak = False
    #

    messages = (
        build_messages(question, hits, cfg, history)
        if not rag_enabled or (hits and not is_weak)
        else None
    )

    return (
        question, rewritten_question, cfg, hits, messages, route_decision, started,
        rewrite_ms, route_ms, retrieve_ms, rerank_ms, rag_enabled, filters,
    )


# keeps the model's Answer section
def _answer_part(answer_text: str, cfg: Config) -> str:
    """8. With prompt v4 the model writes "Evidence: ... Answer: ..."; the student only sees the answer part.

    If the model forgets the "Answer:" line, the whole text is kept, so a correct answer is never lost.
    """
    if cfg.generation.get("prompt_version") != "v4":
        return answer_text
    match = re.search(r"answer\s*:\s*", answer_text, flags=re.IGNORECASE)
    return answer_text[match.end():].strip() if match else answer_text


# if course-table produces exact ranking, uses that instead of asking LLM to calculate
def _ranking_answer(hits: list[Hit]) -> str | None:
    """1. Comparison questions are answered with the ranking line itself, without calling the model.

    2. Even with the ranking written out as S1, the 3B model kept merging the first two groups
       ("Strategies and Management Control, both 40%"; Strategies is 45%). The table already holds
       the right answer, so copying it is exact and also saves the generation time.
    """
    if hits and hits[0].chunk.chunk_id == "course_ranking":
        return f"{hits[0].chunk.text} [{hits[0].label}]"
    return None


# checks if answer is valid from RAG
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
    if invalid_citation or not used_labels or not answer_text:
        return REFUSAL, []
    return answer_text, [hits_by_label[label] for label in used_labels]

# packages the answer into an AnswerResult
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


# for questions naming multiple courses, answers each separately and combines the answers
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


# main flow
# prepares everything -> refuses without prompt -> uses table rankings -> calls LLM -> validates and packages answer
def answer(
    question: str,
    history: list[dict] | None = None,
    filters: Filters | None = None,
    cfg: Config | None = None,
) -> AnswerResult:
    """Generate an answer from retrieved evidence, or run the no-RAG baseline."""
    split = _split_by_course(question, history, filters, cfg or load_config("configs/final.yaml"))
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


# yields answer in pieces as LLM generates it (app can show a streamed answer sooner)
def answer_stream(
    question: str,
    history: list[dict] | None = None,
    filters: Filters | None = None,
    cfg: Config | None = None,
) -> Iterator[str | AnswerResult]:
    """Yield generated text pieces and then the validated final result."""
    split = _split_by_course(question, history, filters, cfg or load_config("configs/final.yaml"))
    if split is not None:
        yield split.answer_text
        yield split
        return
    if (cfg or load_config("configs/final.yaml")).generation.get("prompt_version") == "v4":
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
