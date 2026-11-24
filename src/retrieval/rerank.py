# Rerank the merged search results with a cross-encoder (ms-marco-MiniLM-L-6-v2) and keep the best 5
# (top_k in base.yaml). _prepare_answer() in src/generation/answer.py calls it after the RRF fusion, and
# _course_search_hits() in the same file calls it for the search results added to cross-course questions.
# The searches compare the question and each chunk as two separate vectors, which is fast but approximate.
# A cross-encoder reads the question and the chunk together and scores how well the chunk answers the
# question: it's slower, but more precise, so we only use it on the candidates the searches already found.
# Its scores are also what the self-check (src/generation/selfcheck.py) uses to refuse weak sources.
from __future__ import annotations

# For the exponential of the sigmoid, and to check that a score is a real number.
import math
# replace() gives us a copy of a search result (a Hit) with a few fields changed.
from dataclasses import replace
# lru_cache remembers what a function returned, so the model is only loaded the first time.
from functools import lru_cache


# Load the cross-encoder once and keep it in memory. local_files_only: the model is never downloaded while
# answering, because the system works offline. lru_cache keeps one loaded model per model name (up to 3),
# so every later call gets it straight away.
@lru_cache(maxsize=3)
def _model(model_name: str):
    try:
        # We import the library here and not at the top of the file, so it's only loaded when we really need
        # the model.
        from sentence_transformers import CrossEncoder

        # The model and its tokenizer (which splits the text into tokens) are read from the files that are
        # already on the laptop.
        return CrossEncoder(
            model_name,
            automodel_args={"local_files_only": True},
            tokenizer_args={"local_files_only": True},
        )
    # If the model can't be loaded from the laptop, we stop with a clear message; "from exc" keeps the
    # original error underneath, so we can see what really failed.
    except Exception as exc:
        raise RuntimeError(
            f"Reranker model {model_name!r} is not available locally. "
            "Build/download the model before running offline."
        ) from exc


# Score every candidate against the query and return the top_k best, each with its score between 0 and 1.
# query is the question, hits are the candidates from the RRF fusion (src/retrieval/fusion.py) and cfg is the
# configuration.
def rerank(query: str, hits: list, top_k: int, cfg) -> list:
    # top_k has to be a whole number of at least 1. We also reject True and False, because Python treats
    # them as integers.
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError(f"top_k must be a positive integer; got {top_k!r}")
    # With an empty question or no candidates there's nothing to rerank.
    if not query.strip() or not hits:
        return []
    # The name of the reranker model, from the config (retrieval.reranker_model in base.yaml).
    model_name = cfg.retrieval.get("reranker_model")
    # Without a model name the reranker can't run, so the configuration is wrong.
    if not model_name:
        raise ValueError("retrieval.reranker_model must be configured when reranking is enabled")
    # One (question, chunk) pair per candidate. For the chunk we use embed_text (course name + position +
    # text), the same text the searches used.
    pairs = [
        (query, hit.chunk.embed_text)
        for hit in hits
    ]
    try:
        # The cross-encoder reads each pair and gives it one raw score: the higher the score, the better the
        # chunk answers the question. show_progress_bar=False hides the progress bar.
        scores = _model(model_name).predict(pairs, show_progress_bar=False)
    # If the model fails, we raise the error again with the model name, so it's clear which step failed.
    except Exception as exc:
        raise RuntimeError(f"Could not rerank hits with local model {model_name!r}") from exc
    # There has to be exactly one score per candidate; otherwise scores and chunks would be matched wrongly.
    if len(scores) != len(hits):
        raise ValueError("reranker returned a different number of scores than input hits")
    # The cross-encoder gives raw scores of any size. The sigmoid turns each one into a confidence between
    # 0 and 1, so we can use one fixed threshold (the self-check's 0.2) for every question.
    normalized_scores = []
    for score in scores:
        # float() turns the score into a plain Python number (the model gives numpy numbers).
        score = float(score)
        # An infinite or NaN score means the model gave an invalid result, so it's better to stop.
        if not math.isfinite(score):
            raise ValueError("reranker returned a non-finite score")
        # Two equivalent ways of writing the sigmoid, chosen by the sign, so exp() never overflows.
        if score >= 0:
            # Positive score: 1 / (1 + e^(-score)). Here e^(-score) is at most 1, so it can't overflow.
            confidence = 1.0 / (1.0 + math.exp(-score))
        else:
            # Negative score: e^score / (1 + e^score), the same value written another way. Here e^score is
            # below 1, so it can't overflow either.
            exp_score = math.exp(score)
            confidence = exp_score / (1.0 + exp_score)
        # The confidences are saved in the same order as the candidates.
        normalized_scores.append(confidence)
    # Highest confidence first; equal scores are ordered by id, so the order is always the same.
    # zip() pairs each confidence with its candidate, and the minus sign sorts from the highest to the lowest.
    ranked = sorted(
        zip(normalized_scores, hits),
        key=lambda item: (-item[0], item[1].chunk.chunk_id),
    )
    # We keep the best top_k, numbered from 1, with their reranker score. replace() copies each candidate
    # with its new position (rank) and its confidence (score_rerank); the other fields, such as the scores of
    # the two searches and of the fusion, stay as they were. score_rerank is what the self-check reads and
    # what the app shows in its "Behind the scenes" panel (app_components/behind.py).
    return [
        replace(hit, rank=rank, score_rerank=score)
        for rank, (score, hit) in enumerate(ranked[:top_k], start=1)
    ]


# Load the cross-encoder when the app starts (src/warmup.py calls it), so the first answer isn't slower.
def warm_up(cfg) -> None:
    # It's only needed when reranking is switched on (retrieval.rerank).
    if not cfg.retrieval.get("rerank", False):
        return
    # The model name has to be in the config, as in rerank() above.
    model_name = cfg.retrieval.get("reranker_model")
    if not model_name:
        raise ValueError("retrieval.reranker_model must be configured when reranking is enabled")
    # Loading the model here leaves it in the lru_cache memory, so rerank() later finds it already loaded.
    _model(model_name)
