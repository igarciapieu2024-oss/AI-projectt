# This contains the answer phase, where it takes a student's question and turns it into an answer with citations.
# The evaluation (eval/run_eval.py) calls answer(), and the app (app_components/answer_view.py) calls
# answer_stream(), which shows the answer while it is being written. Both go through the same steps:
#   1. prepare the question (rewrite.py)          2. decide its path (router.py)
#   3. choose where to search (course filter, per-course answers, course table)
#   4. search ChromaDB and BM25 and merge them    5. rerank and run the self-check
#   6. build the prompt (prompts.py)              7. ask the LLM (llm.py) and check the citations
# Steps 1 to 6 live in _prepare_answer(), so both functions share them. The settings come from the config
# (configs/final_system.yaml by default). In the end both return an AnswerResult (src/schemas.py), which the
# app shows and the evaluation saves to calculate the metrics (eval/metrics.py).
from __future__ import annotations

# We copy the config when we answer course by course (see _split_by_course() below).
import copy
# Regular expressions, to find the citations and the ends of the sentences in an answer.
import re
# To measure how long each step takes; the app shows these times and the evaluation saves them.
import time
# replace() gives us a copy of a search result (a Hit) with a few fields changed
from dataclasses import replace
# answer_stream() returns an Iterator, because it hands out the answer piece by piece.
from typing import Iterator

# The configuration, and the function that reads it from a YAML file in configs/.
from src.config import Config, load_config
# The other parts of the answer phase: rewriting and translating the question, the self-check and the course table.
from src.generation import rewrite, selfcheck, table_qa
# chat() is how we talk to the local LLM, through Ollama.
from src.generation.llm import chat
# build_messages() writes the prompt: the rules, the conversation so far and the question with its sources.
from src.generation.prompts import build_messages
# The search side: dense search (ChromaDB), keyword search (BM25), the RRF fusion and the reranker.
from src.retrieval import bm25, dense, fusion, rerank
# Tells us which courses a question mentions.
from src.retrieval.filters import mentioned_course_codes
# The router decides which path each question takes. We import it as route_question because we also use a
# variable called route further down.
from src.retrieval.router import route as route_question
# The shared formats from src/schemas.py: the final result, a citation, the search filters, a search result,
# the refusal message and the router's decision.
from src.schemas import AnswerResult, Citation, Filters, Hit, REFUSAL, RouteDecision

# How the model writes a citation: an S and a number inside square brackets or parentheses, like [S1] or (s2).
_CITATION = re.compile(r"[\[(]S(\d+)[\])]", re.IGNORECASE)
# Where a sentence ends, counting the citations that can come just before or just after the full stop.
# [.!?] is the final punctuation, and (?=\s|$) says it has to be followed by a space or by the end of the
# text, so a dot inside a number doesn't count as the end of a sentence.
_COMPLETE_CITED_SENTENCE = re.compile(
    r"(?:\s*\[S\d+\])*[.!?](?:\s*\[S\d+\])*(?=\s|$)",
    re.IGNORECASE,
)
# The same idea without citations, for answers written without RAG.
# answer_stream() also uses both patterns to send the answer to the app one sentence at a time.
_COMPLETE_SENTENCE = re.compile(r"[.!?](?=\s|$)")


# If the model hits the token limit (done_reason "length"), its last sentence is usually cut in half. We drop
# that unfinished sentence, so the student never sees half a sentence. If not even one sentence was finished
# we return "", and with RAG the citation check then turns it into the refusal message.
def _trim_to_last_complete_sentence(answer_text: str, rag_enabled: bool) -> str:
    # With RAG we only count a sentence as finished once its citations are there, so we never cut a citation.
    sentence_pattern = _COMPLETE_CITED_SENTENCE if rag_enabled else _COMPLETE_SENTENCE
    # Find every place in the text where a sentence ends.
    completed = list(sentence_pattern.finditer(answer_text))
    # Not a single finished sentence, so there is nothing we can show.
    if not completed:
        return ""
    # Keep everything up to the end of the last finished sentence.
    return answer_text[:completed[-1].end()].rstrip()


# How many sources the LLM gets (retrieval.top_k, 5 in base.yaml).
def _top_k(cfg: Config) -> int:
    # If the config doesn't say, we use 5.
    top_k = cfg.retrieval.get("top_k", 5)
    # It has to be a whole number of at least 1. We also reject True and False, because Python treats them
    # as integers.
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError(f"retrieval.top_k must be a positive integer; got {top_k!r}")
    return top_k


# The search itself: the dense search (ChromaDB) and BM25, merged with RRF (fusion.py). If retrieval.mode is
# "dense", only the dense search runs, which is how our first version worked. What comes out are the merged
# candidates, which the reranker then narrows down to the best 5.
def _build_hits(question: str, cfg: Config, filters: Filters | None) -> list[Hit]:
    top_k = _top_k(cfg)
    # We use both searches ("hybrid") unless the config asks for something else.
    mode = cfg.retrieval.get("mode", "hybrid")
    # Only these two modes exist, so anything else is a typo in the config.
    if mode not in {"dense", "hybrid"}:
        raise ValueError(f"unsupported retrieval.mode: {mode!r}")
    # How many candidates each search brings back (k_dense and k_bm25, 20 in base.yaml); top_k if not set.
    dense_k = cfg.retrieval.get("k_dense", top_k)
    # The same check as for top_k.
    if not isinstance(dense_k, int) or isinstance(dense_k, bool) or dense_k < 1:
        raise ValueError(f"retrieval.k_dense must be a positive integer; got {dense_k!r}")

    # Dense search: the question becomes a vector and ChromaDB gives back the dense_k closest chunks
    # (src/retrieval/dense.py). If a course was chosen, filters keeps the search inside that course.
    dense_hits = dense.search(question, cfg, filters, dense_k)
    # In "dense" mode these are already our candidates.
    if mode == "dense":
        return dense_hits

    bm25_k = cfg.retrieval.get("k_bm25", top_k)
    # The same check as for top_k.
    if not isinstance(bm25_k, int) or isinstance(bm25_k, bool) or bm25_k < 1:
        raise ValueError(f"retrieval.k_bm25 must be a positive integer; got {bm25_k!r}")
    # Keyword search: BM25 gives back the bm25_k chunks that best match the words of the question
    # (src/retrieval/bm25.py), with the same filters.
    bm25_hits = bm25.search(question, cfg, filters, bm25_k)
    # The RRF constant (60 in base.yaml).
    rrf_k = cfg.retrieval.get("rrf_k", 60)
    # The same check as for top_k.
    if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k < 1:
        raise ValueError(f"retrieval.rrf_k must be a positive integer; got {rrf_k!r}")
    # Merge the two lists into one with RRF. RRF looks at the position of each chunk in each list, not at its
    # score, because the two searches score on completely different scales.
    return fusion.rrf([dense_hits, bm25_hits], k=rrf_k)


# Extra search results for cross-course questions, on top of the course table rows. The table has the
# weights, attendance, professor and sessions, but not details like the session of a midterm or the
# textbook, and questions like that were refused when the rows were the only sources.
# Each course in the rows gets its best 1 or 2 search results, searched only in that course; when all 7
# rows are there (no course named), we add the 3 best results from the whole knowledge base instead.
def _course_search_hits(question: str, cfg: Config, table_rows: list[Hit]) -> list[Hit]:
    # The course codes in the table rows without repeats; dict.fromkeys() removes duplicates but keeps the order.
    codes = list(dict.fromkeys(hit.chunk.course_code for hit in table_rows if hit.chunk.course_code))
    # Whether the reranker is switched on (retrieval.rerank).
    use_rerank = cfg.retrieval.get("rerank", False)

    # A small helper: the best `keep` results of one search. Only the 10 best fused candidates go to the
    # reranker, so this extra search doesn't add much time.
    def best(filters: Filters | None, keep: int) -> list[Hit]:
        # Run both searches with RRF and take the first 10 candidates.
        candidates = _build_hits(question, cfg, filters)[:10]
        # Nothing found, so nothing to add.
        if not candidates:
            return []
        # With the reranker we keep its best `keep`; without it, simply the first `keep` from the fusion.
        return rerank.rerank(question, candidates, keep, cfg) if use_rerank else candidates[:keep]

    # The rows cover all 7 courses (the question didn't name any) or have no course code: we add the 3 best
    # results from the whole knowledge base, with no filter.
    if not codes or len(codes) >= 7:
        return best(None, 3)
    # 2 results per course for one or two courses, 1 when there are more, so the prompt doesn't get too long.
    keep = 2 if len(codes) <= 2 else 1
    # Search each course separately, filtered to that course, and collect everything.
    found = []
    for code in codes:
        found.extend(best(Filters(course_code=code), keep))
    return found


# Everything that happens before the LLM: prepare the question, route it, pick the filters, search, rerank,
# run the self-check and build the prompt. It returns the original and the rewritten question, the config,
# the sources (hits), the messages for the LLM (None when the answer has to be a refusal), the route, the
# start time, the time of each step, whether RAG is on and the filters it used. Both answer() and
# answer_stream() start by calling it.
def _prepare_answer(
    question: str,
    history: list[dict] | None,
    filters: Filters | None,
    cfg: Config | None,
) -> tuple[
    str, str, Config, list[Hit], list[dict[str, str]] | None, RouteDecision,
    float, float, float, float, float, bool, Filters | None,
]:
    # Clean the spaces around the question; an empty question can't be answered.
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")

    # If no config is given, we use the final system.
    cfg = cfg or load_config("configs/final_system.yaml")
    # Start the clock for the total time of the answer.
    started = time.perf_counter()
    # Step 1: prepare the question. The rewritten question is the one we route and search with, but the LLM
    # still gets the original question and the conversation.
    # To start with, the question stays as it is.
    rewritten_question = question
    # rag is only false in baseline B0, where the LLM answers on its own, with no sources.
    rag_enabled = cfg.generation.get("rag", True)
    # This stays at 0 if nothing gets rewritten or translated.
    rewrite_ms = 0.0
    # We only translate when RAG is on and generation.translate_query is true.
    translation_enabled = rag_enabled and cfg.generation.get("translate_query", False)
    # There is only work to do if there is a conversation and rewriting is on (generation.rewrite), or if
    # translation is on.
    if (history and cfg.generation.get("rewrite", False)) or translation_enabled:
        rewrite_started = time.perf_counter()
        # We translate only if translation is on and the question isn't in English (langid detects the language).
        translate_to_english = translation_enabled and rewrite.detect_language(question) != "en"
        # A follow-up is rewritten using the conversation (and translated in the same call if needed); a new
        # question is only translated, and only if it isn't in English.
        if history and cfg.generation.get("rewrite", False):
            if translate_to_english:
                # Rewrite and translate in a single call to the LLM.
                rewritten_question = rewrite.rewrite(
                    question,
                    history,
                    cfg,
                    translate_to_english=True,
                )
            else:
                # Just rewrite. If the question isn't really a follow-up, rewrite() gives it back unchanged.
                rewritten_question = rewrite.rewrite(question, history, cfg)
        # No conversation (or rewriting is off): we only translate.
        elif translate_to_english:
            rewritten_question = rewrite.translate_to_english(question, cfg)
        # How long this step took, in milliseconds.
        rewrite_ms = (time.perf_counter() - rewrite_started) * 1000

    # Step 2: route the question, with the classifier, or with the LLM when the classifier isn't sure.
    # Without RAG there is no routing, so these default values stay.
    route_ms = 0.0
    route_decision = RouteDecision(label="none", prob=0.0, source="none")
    if rag_enabled:
        route_started = time.perf_counter()
        # The router reads the rewritten question and returns one of the 4 labels (course_admin,
        # course_content, cross_course, out_of_scope), how sure it is and who made the decision
        # (src/retrieval/router.py).
        route_decision = route_question(rewritten_question, cfg)
        route_ms = (time.perf_counter() - route_started) * 1000

    # Step 3: choose where to search.
    # With course_split, a question that compares two or more named courses on something the table has goes
    # to the course table, even if the router chose another label. When we searched these normally, the model
    # mixed up the weights of the two courses.
    # In order, the conditions are: RAG is on, course_split is on, the router hasn't already sent the question
    # to the table or out of scope, the question names at least two courses, and it asks to rank or compare
    # them (asks_for_ranking() in table_qa.py).
    if (
        rag_enabled
        and cfg.generation.get("course_split", False)
        and route_decision.label not in ("cross_course", "out_of_scope")
        and len(_named_courses(rewritten_question)) >= 2
        and table_qa.asks_for_ranking(rewritten_question)
    ):
        # We send it to the table, with source "course_split" so we can see later that this rule decided it.
        route_decision = RouteDecision(label="cross_course", prob=1.0, source="course_split")

    # With routing.named_course_in_scope (tried in experiment 09, not in the final system), a question that
    # names a course is never sent out of scope by the router: it gets searched, and the self-check or the
    # model can still refuse it if the material doesn't have the answer. We didn't keep it, because it made
    # the system answer a private question it should have refused.
    # getattr(cfg, "routing", {}) means this also works with a config that has no routing section.
    if (
        rag_enabled
        and route_decision.label == "out_of_scope"
        and getattr(cfg, "routing", {}).get("named_course_in_scope", False)
        and _named_courses(rewritten_question)
    ):
        route_decision = RouteDecision(label="course_content", prob=route_decision.prob, source="named_course")

    # Out-of-scope questions are refused right here, without searching. With routing.out_of_scope_search
    # (tried in experiments 06 to 08, not in the final system) they were searched anyway and only the
    # self-check could refuse them, because the router had sent two answerable questions out of scope and
    # nothing could fix that.
    if (
        rag_enabled
        and route_decision.label == "out_of_scope"
        and not getattr(cfg, "routing", {}).get("out_of_scope_search", False)
    ):
        # No sources and no prompt (None): answer() and answer_stream() turn this into the refusal message.
        # The search and rerank times are 0 because we never searched.
        return (
            question, rewritten_question, cfg, [], None, route_decision, started,
            rewrite_ms, route_ms, 0.0, 0.0, rag_enabled, filters,
        )

    # With retrieval.auto_course_filter (final_system.yaml), a question that names exactly one course is only
    # searched in that course, so slides from other courses never reach the model.
    # We skip this if a course was already chosen (in the app, or by the per-course answers), and for
    # cross-course questions, which go to the course table anyway.
    if (
        rag_enabled
        and cfg.retrieval.get("auto_course_filter", False)
        and not (filters and filters.course_code)
        and route_decision.label != "cross_course"
    ):
        # The courses the question names.
        named = _named_courses(rewritten_question)
        if len(named) == 1:
            # Filter to that course, keeping the sessions of the previous filter if there was one.
            filters = Filters(course_code=named[0], sessions=list(filters.sessions) if filters else [])

    # Step 4: search. table_hits tells us whether the sources come from the course table.
    retrieval_started = time.perf_counter()
    table_hits = False
    hits = []
    if rag_enabled:
        # Cross-course questions are answered from the course table, plus some search results when the rows
        # aren't a ranking. If the question is already limited to one course (chosen in the app or by the
        # per-course answers), we search normally instead.
        if route_decision.label == "cross_course" and not (filters and filters.course_code):
            # One source per course row, or the ranking line first (src/generation/table_qa.py).
            hits = table_qa.answer_from_table(rewritten_question, cfg)
            # Empty, for example, when there is no course table; then the normal search below takes over.
            table_hits = bool(hits)
            # If the first source isn't the ranking line, we add some search results to the rows, because the
            # table doesn't have every detail.
            if table_hits and hits[0].chunk.chunk_id != "course_ranking":
                hits = hits + _course_search_hits(rewritten_question, cfg, hits)
        # Every other question, or a table that gave no rows: the normal search.
        if not hits:
            hits = _build_hits(rewritten_question, cfg, filters)
    # How long the search took, in milliseconds.
    retrieve_ms = (time.perf_counter() - retrieval_started) * 1000
    # Step 5: rerank the results down to the best top_k and run the self-check. We don't rerank table rows:
    # they are already exactly the rows the question needs.
    rerank_ms = 0.0
    if rag_enabled and hits and not table_hits:
        top_k = _top_k(cfg)
        if cfg.retrieval.get("rerank", False):
            rerank_started = time.perf_counter()
            # The cross-encoder reads each candidate together with the question and keeps the best top_k
            # (src/retrieval/rerank.py).
            hits = rerank.rerank(rewritten_question, hits, top_k, cfg)
            rerank_ms = (time.perf_counter() - rerank_started) * 1000
        else:
            # Without the reranker (like our first version), we just keep the first top_k from the fusion.
            hits = hits[:top_k]
        # The self-check (generation.selfcheck): True when even the best source scores below the threshold
        # (0.2). In that case no prompt is built below (src/generation/selfcheck.py).
        is_weak = cfg.generation.get("selfcheck", False) and selfcheck.is_weak(hits, cfg)
        # Number the sources S1, S2... in their final order. These are the labels the model has to cite.
        hits = [replace(hit, rank=rank, label=f"S{rank}") for rank, hit in enumerate(hits, start=1)]
    elif table_hits:
        # Table rows don't have a reranker score, so we don't run the self-check on them.
        is_weak = False
        # The table sources get their S1, S2... labels too.
        hits = [replace(hit, rank=rank, label=f"S{rank}") for rank, hit in enumerate(hits, start=1)]
    else:
        # No RAG, or no sources at all: nothing to check.
        is_weak = False
    # Step 6: build the prompt, unless there are no sources or they're too weak (then None, which means refuse).
    # Without RAG (baseline B0) we always build the prompt, just without sources.
    messages = (
        build_messages(question, hits, cfg, history)
        if not rag_enabled or (hits and not is_weak)
        else None
    )
    # Hand back everything the next steps need, in the same order answer() and answer_stream() unpack it.
    return (
        question, rewritten_question, cfg, hits, messages, route_decision, started,
        rewrite_ms, route_ms, retrieve_ms, rerank_ms, rag_enabled, filters,
    )


# With prompt v4 (experiment 08) the model writes "Evidence: ... Answer: ...", and the student should only
# see what comes after "Answer:". If the model forgets that line we keep the whole text, so we never lose a
# correct answer. With the other prompt versions the text comes back unchanged.
def _answer_part(answer_text: str, cfg: Config) -> str:
    # Only prompt v4 asks for the "Evidence" and "Answer" parts.
    if cfg.generation.get("prompt_version") != "v4":
        return answer_text
    # Look for "Answer:" in upper or lower case, with or without spaces around the colon.
    match = re.search(r"answer\s*:\s*", answer_text, flags=re.IGNORECASE)
    # Return what comes after "Answer:", or the whole text if it isn't there.
    return answer_text[match.end():].strip() if match else answer_text


# Ranking questions are answered with the ranking line itself (built by table_qa.py), without calling the
# model. Even with the ranking written out as S1, the small model kept merging the first two groups. The
# table already has the right answer, so copying it is exact and also saves the generation time.
# Returns None when the first source isn't a ranking line.
def _ranking_answer(hits: list[Hit]) -> str | None:
    # When there is a ranking line, it's the first source, with the id "course_ranking".
    if hits and hits[0].chunk.chunk_id == "course_ranking":
        # The answer is the ranking line plus its label, so it's cited like any other answer.
        return f"{hits[0].chunk.text} [{hits[0].label}]"
    return None


# One cited sentence of the answer: body is the text, cites is one or more [S..] labels after it, and end is
# the final punctuation, if there is any.
_CITED_PART = re.compile(r"(?P<body>[^\[]*?)(?P<cites>(?:\s*\[S\d+\])+)(?P<end>\s*[.!?]?)")


# When several sources say the same thing, the model sometimes writes the same sentence once per source,
# each time with a different citation. Here we write that sentence only once, with all its citations, so no
# fact and no citation gets lost. Answers without an exactly repeated sentence come back unchanged. This only
# changes how the answer looks, not what it says.
def _merge_repeated_sentences(answer_text: str) -> str:
    # Split the answer into its cited sentences, skipping empty matches.
    parts = [m for m in _CITED_PART.finditer(answer_text) if m.group(0)]
    # To compare the sentences we use lower case, single spaces and no final punctuation.
    keys = [" ".join(m.group("body").lower().split()).strip(" .,;:!?") for m in parts]
    # If every sentence is different, there's nothing to merge.
    if len({k for k in keys if k}) == len([k for k in keys if k]):
        return answer_text  # nothing repeated
    # For each sentence, collect every label it was cited with, without repeating any.
    merged: dict[str, list[str]] = {}  # sentence -> its labels, in order of appearance
    for match, key in zip(parts, keys):
        if key:
            labels = merged.setdefault(key, [])
            labels += [c for c in re.findall(r"\[S\d+\]", match.group("cites")) if c not in labels]
    # Now we rebuild the answer, writing each sentence the first time it appears, with all its citations.
    pieces, written = [], set()
    for match, key in zip(parts, keys):
        # A piece with no real text (only citations or punctuation) stays as it was.
        if not key:
            pieces.append(match.group(0))
        # The first copy of a sentence goes in with all its labels; later copies are skipped.
        elif key not in written:
            written.add(key)
            pieces.append(match.group("body").rstrip() + " " + "".join(merged[key]) + match.group("end"))
    # Anything after the last cited sentence stays as it is.
    tail = answer_text[parts[-1].end():] if parts else ""
    return ("".join(pieces) + tail).strip()


# The citation check, the last part of step 7. Every [S..] in the answer has to be one of the sources the
# model actually received. If the answer invents a label, or has no citation at all, we replace it with the
# refusal message, so the student never sees a claim that can't be traced back to a source. It returns the
# answer and the sources it cites. Without RAG there is nothing to check.
def _validated_answer(answer_text: str, hits: list[Hit], rag_enabled: bool) -> tuple[str, list[Hit]]:
    # No RAG means no sources, so the answer goes back as it is, with no cited sources.
    if not rag_enabled:
        return answer_text.strip(), []

    # Each label (S1, S2...) with its source, so we can check the citations against them.
    hits_by_label = {hit.label: hit for hit in hits}
    # The labels the answer really cites, in the order they first appear.
    used_labels: list[str] = []
    # Switches to True if the answer cites a label that isn't one of the sources.
    invalid_citation = False

    # For each citation: write it as [S1] (it may come as "(S1)" or "[s1]"), remember the label, and flag it
    # if it isn't one of the sources.
    def normalize_citation(match: re.Match[str]) -> str:
        # nonlocal lets this inner function change invalid_citation, which belongs to _validated_answer().
        nonlocal invalid_citation
        # The label in the same form as our sources: S plus the number.
        label = f"S{match.group(1)}"
        # Not one of the sources: flag it and leave the text untouched.
        if label not in hits_by_label:
            invalid_citation = True
            return match.group(0)
        # A valid label is remembered only once.
        if label not in used_labels:
            used_labels.append(label)
        # All valid citations end up written the same way: square brackets and a capital S.
        return f"[{label}]"

    # Run normalize_citation() on every citation in the answer.
    answer_text = _CITATION.sub(normalize_citation, answer_text).strip()
    # If a sentence was repeated once per source, we write it only once, with all its citations.
    answer_text = _merge_repeated_sentences(answer_text)
    # An invented label, no citation at all, or an empty answer: the student gets the refusal message.
    if invalid_citation or not used_labels or not answer_text:
        return REFUSAL, []
    # The answer, and the sources it cites in the order they are first cited.
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
    # First the citation check: the final text of the answer and the sources it really cites.
    answer_text, cited_hits = _validated_answer(answer_text, hits, rag_enabled)
    # One Citation per cited source: its label, its chunk and the original slides or syllabus sections that
    # chunk covers. The evaluation checks each citation against the sources (eval/metrics.py).
    citations = [
        Citation(label=hit.label, chunk_id=hit.chunk.chunk_id, covers=hit.chunk.covers)
        for hit in cited_hits
    ]
    return AnswerResult(
        question=question,
        rewritten_question=rewritten_question,
        route=route_decision,
        # The filters we actually used, or an empty Filters if no course was chosen.
        filters_used=filters or Filters(),
        # All the sources the model was given, S1, S2...
        context_hits=hits,
        answer_text=answer_text,
        citations=citations,
        # True when the final answer is the refusal message; the evaluation metrics use it.
        refused=rag_enabled and answer_text == REFUSAL,
        # The time of each step in milliseconds, rounded to one decimal; total counts from the very start of
        # the answer. The app shows these times in its "Behind the scenes" panel.
        timings_ms={
            "rewrite": round(rewrite_ms, 1),
            "route": round(route_ms, 1),
            "retrieve": round(retrieve_ms, 1),
            "rerank": round(rerank_ms, 1),
            "generate": round(generate_ms, 1),
            "total": round((time.perf_counter() - started) * 1000, 1),
        },
        # The config and the knowledge base we used, so every saved answer says which version of the system
        # wrote it.
        config_name=cfg.name,
        kb_sig=cfg.kb_sig,
    )


# Short names students use for two of the courses. We only add them when we look for the courses a question
# names (the course filter and the per-course answers), so the matching of the versions we evaluated earlier
# stays the same.
_COURSE_ALIASES = [
    {"course_code": "strategies", "course_name": "Strategy"},
    {"course_code": "ai_ml", "course_name": "Machine Learning Foundations"},
]


# The codes of the courses a question names, each only once, using the names in the course table plus the
# short names above.
def _named_courses(question: str) -> list[str]:
    # mentioned_course_codes() (src/retrieval/filters.py) finds every course whose code or name appears in the
    # question, and dict.fromkeys() removes repeats while keeping the order.
    return list(dict.fromkeys(mentioned_course_codes(question, table_qa.course_rows() + _COURSE_ALIASES)))


# With generation.course_split (final_system.yaml), a question that names two or more courses is answered
# course by course. When the model had the sources of several courses together, it gave one course's fact for
# all of them. So each course is searched and answered on its own, with only its own sources, and then the
# parts are joined with their labels renumbered, so every citation still points to the right source. Ranking
# questions aren't split: they keep the ranking line of the course table. Returns None when the question
# isn't split.
def _split_by_course(
    question: str,
    history: list[dict] | None,
    filters: Filters | None,
    cfg: Config,
) -> AnswerResult | None:
    # Only when course_split is on, and only with RAG (without sources there is nothing to split).
    if not cfg.generation.get("course_split", False) or not cfg.generation.get("rag", True):
        return None
    # A course was already chosen (in the app, or because this already is one of the per-course answers),
    # so we don't split again.
    if filters and filters.course_code:
        return None
    # The course table rows, to get the name of each course.
    rows = table_qa.course_rows()
    # The courses the question names.
    codes = _named_courses(question)
    # Fewer than two courses, or a ranking question: we answer it normally.
    if len(codes) < 2 or table_qa.asks_for_ranking(question):
        return None
    # The full name of each course, used in the question we ask for each course and in the joined answer.
    names = {row["course_code"]: row["course_name"] for row in rows}
    started = time.perf_counter()
    # A copy of the config with course_split switched off, so each per-course answer isn't split again.
    single = copy.copy(cfg)
    single.generation = {**cfg.generation, "course_split": False}
    # parts: the answer for each course; hits: all the sources, renumbered; timings: the time of each step.
    parts, hits, timings = [], [], {}
    for code in codes:
        # We ask the question for this course only: we add which course to answer for, and the filter keeps
        # the search inside that course (names.get() uses the code if the course isn't in the table).
        sub = answer(
            f"{question} Answer only for {names.get(code, code)}.",
            history,
            Filters(course_code=code, sessions=list(filters.sessions) if filters else []),
            single,
        )
        # Number this course's sources after the ones we already have, and shift its citations the same way.
        offset = len(hits)
        hits.extend(replace(hit, rank=offset + hit.rank, label=f"S{offset + hit.rank}") for hit in sub.context_hits)
        # If this course's answer was refused, we write a short sentence instead, so the other courses are
        # still answered.
        if sub.refused:
            parts.append(f"{names.get(code, code)}: the material does not cover it.")
        # Otherwise, its answer with offset added to every citation number: if the first course had 5 sources,
        # [S1] of the second course becomes [S6].
        else:
            parts.append(re.sub(r"\[S(\d+)\]", lambda m: f"[S{int(m.group(1)) + offset}]", sub.answer_text))
        # Add up the time of each step over all the courses.
        for step, ms in sub.timings_ms.items():
            timings[step] = timings.get(step, 0.0) + ms
    # The joined answer counts as cross_course, decided by course_split.
    route = RouteDecision(label="cross_course", prob=1.0, source="course_split")
    # Join the parts into one answer; _make_result() checks its citations against all the sources.
    return _make_result(
        question, question, route, filters, cfg, hits, " ".join(parts), started,
        timings.get("rewrite", 0.0), timings.get("route", 0.0), timings.get("retrieve", 0.0),
        timings.get("rerank", 0.0), timings.get("generate", 0.0), True,
    )


# Answer a question and return the full AnswerResult. The evaluation uses this one (eval/run_eval.py).
# history is the conversation so far (for follow-ups), filters can force a course, and cfg is the
# configuration (final_system.yaml if none is given). Without RAG (baseline B0) the LLM answers on its own.
def answer(
    question: str,
    history: list[dict] | None = None,
    filters: Filters | None = None,
    cfg: Config | None = None,
) -> AnswerResult:
    # If the question names two or more courses (and course_split is on), we answer it course by course.
    split = _split_by_course(question, history, filters, cfg or load_config("configs/final_system.yaml"))
    if split is not None:
        return split
    # Steps 1 to 6, shared with answer_stream().
    (
        question, rewritten_question, cfg, hits, messages, route_decision, started,
        rewrite_ms, route_ms, retrieve_ms, rerank_ms, rag_enabled, filters,
    ) = _prepare_answer(
        question, history, filters, cfg
    )
    # No prompt means there were no sources or they were too weak: the answer is the refusal message, and the
    # LLM isn't called (so the generation time is 0).
    if rag_enabled and messages is None:
        return _make_result(
            question, rewritten_question, route_decision, filters, cfg, hits, REFUSAL, started,
            rewrite_ms, route_ms, retrieve_ms, rerank_ms, 0.0, rag_enabled,
        )
    # Without RAG there is always a prompt, so if it's missing, something went wrong in _prepare_answer().
    if messages is None:
        raise RuntimeError("answer prompt was not constructed")

    # Ranking questions: the ranking line is the answer, no LLM needed.
    ranking_text = _ranking_answer(hits) if rag_enabled else None
    if ranking_text:
        return _make_result(
            question, rewritten_question, route_decision, filters, cfg, hits, ranking_text, started,
            rewrite_ms, route_ms, retrieve_ms, rerank_ms, 0.0, rag_enabled,
        )

    # Step 7: ask the LLM.
    generation_started = time.perf_counter()
    # Send the prompt to the LLM and wait for the full answer (src/generation/llm.py).
    response = chat(messages, cfg)
    # Without streaming, chat() has to give back text.
    if not isinstance(response, str):
        raise TypeError("chat() must return text when stream=False")
    # The text of the answer. response also carries done_reason, which tells us why the model stopped.
    answer_text = str(response)
    # If the answer was cut by the token limit, it loses its unfinished last sentence. done_reason is "length"
    # when the model stopped because it reached the limit (generation.num_predict, 400 tokens in base.yaml).
    if getattr(response, "done_reason", None) == "length":
        answer_text = _trim_to_last_complete_sentence(answer_text, rag_enabled)
    # An empty answer that wasn't cut means something went wrong in Ollama.
    if not answer_text.strip() and getattr(response, "done_reason", None) != "length":
        raise ValueError("chat() returned empty answer text")
    # How long the LLM took, in milliseconds.
    generate_ms = (time.perf_counter() - generation_started) * 1000
    # With prompt v4 we keep only the part after "Answer:"; otherwise the text stays the same.
    answer_text = _answer_part(answer_text, cfg)
    # _make_result() checks the citations and builds the final result.
    return _make_result(
        question, rewritten_question, route_decision, filters, cfg, hits, answer_text, started,
        rewrite_ms, route_ms, retrieve_ms, rerank_ms, generate_ms, rag_enabled,
    )


# The same steps as answer(), but for the app: it hands out the answer in pieces while the LLM is writing it,
# and the full AnswerResult at the end. A piece only goes out once a sentence (with its citations) is
# complete, so the student never sees a half-written citation.
def answer_stream(
    question: str,
    history: list[dict] | None = None,
    filters: Filters | None = None,
    cfg: Config | None = None,
) -> Iterator[str | AnswerResult]:
    # Per-course answers aren't streamed: we join their parts first and send everything at once.
    split = _split_by_course(question, history, filters, cfg or load_config("configs/final_system.yaml"))
    if split is not None:
        # First the whole text, then the result, which is what the app waits for at the end.
        yield split.answer_text
        yield split
        return
    # With prompt v4 the evidence part must stay hidden, so we let answer() produce the whole answer first.
    if (cfg or load_config("configs/final_system.yaml")).generation.get("prompt_version") == "v4":
        result = answer(question, history, filters, cfg)
        yield result.answer_text
        yield result
        return
    # Steps 1 to 6, the same as in answer().
    (
        question, rewritten_question, cfg, hits, messages, route_decision, started,
        rewrite_ms, route_ms, retrieve_ms, rerank_ms, rag_enabled, filters,
    ) = _prepare_answer(
        question, history, filters, cfg
    )
    # No prompt: we send the refusal message and then the result, without calling the LLM.
    if rag_enabled and messages is None:
        result = _make_result(
            question, rewritten_question, route_decision, filters, cfg, hits, REFUSAL, started,
            rewrite_ms, route_ms, retrieve_ms, rerank_ms, 0.0, rag_enabled,
        )
        yield result.answer_text
        yield result
        return
    # Without RAG there is always a prompt, so if it's missing, something went wrong in _prepare_answer().
    if messages is None:
        raise RuntimeError("answer prompt was not constructed")

    # Ranking questions: we send the ranking line straight away, no LLM needed.
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
    # Step 7 with streaming: the LLM sends its answer in small pieces as it writes it.
    response = chat(messages, cfg, stream=True)
    # With streaming, chat() has to give us the pieces one by one, not a finished text.
    if isinstance(response, str):
        raise TypeError("chat() must return an iterator when stream=True")
    # pieces: everything we've received; pending: text we've received but not shown yet, because its sentence
    # isn't finished.
    pieces: list[str] = []
    pending = ""
    for piece in response:
        # We keep every piece, so we can put the whole answer together at the end.
        pieces.append(piece)
        pending += piece
        # Look for finished sentences in the text we haven't shown yet (with their citations when RAG is on).
        sentence_pattern = _COMPLETE_CITED_SENTENCE if rag_enabled else _COMPLETE_SENTENCE
        completed = list(sentence_pattern.finditer(pending))
        if completed:
            # Send everything up to the end of the last finished sentence, and keep the rest for later.
            end = completed[-1].end()
            yield pending[:end]
            pending = pending[end:]
    # Send what's left, unless the answer was cut by the token limit (then we drop the unfinished part).
    done_reason = getattr(response, "done_reason", None)
    if done_reason != "length" and pending:
        yield pending
    # The whole answer, just as answer() would have received it.
    answer_text = "".join(pieces)
    # A cut answer loses its unfinished last sentence, as in answer().
    if done_reason == "length":
        answer_text = _trim_to_last_complete_sentence(answer_text, rag_enabled)
    # An empty answer that wasn't cut means something went wrong in Ollama.
    if not answer_text.strip() and done_reason != "length":
        raise ValueError("chat() returned no answer text while streaming")
    # How long the LLM took, in milliseconds.
    generate_ms = (time.perf_counter() - generation_started) * 1000
    # Check the citations and build the result, as in answer().
    result = _make_result(
        question, rewritten_question, route_decision, filters, cfg, hits, answer_text, started,
        rewrite_ms, route_ms, retrieve_ms, rerank_ms, generate_ms, rag_enabled,
    )
    # If nothing was shown (for example, every sentence was cut), we show the final text, and then send the
    # result. The app replaces what it streamed with result.answer_text, which has passed the citation check.
    if not answer_text.strip():
        yield result.answer_text
    yield result
