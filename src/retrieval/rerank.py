# Rerank the merged search results with a cross-encoder (ms-marco-MiniLM-L-6-v2) and keep the best 5
# (top_k in base.yaml). Called by _prepare_answer() in src/generation/answer.py after the RRF fusion.
# The searches compare the question and each chunk as two separate vectors, which is fast but approximate.
# A cross-encoder reads the question and the chunk together and scores how well the chunk answers the
# question: slower, but more precise, so it is only used on the candidates the searches already found.
# Its scores are also what the self-check (src/generation/selfcheck.py) uses to refuse weak sources.
from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache


# Load the cross-encoder once and keep it in memory. local_files_only: the model is never downloaded
# while answering, because the system works offline.
@lru_cache(maxsize=3)
def _model(model_name: str):
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(
            model_name,
            automodel_args={"local_files_only": True},
            tokenizer_args={"local_files_only": True},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Reranker model {model_name!r} is not available locally. "
            "Build/download the model before running offline."
        ) from exc


# Score every candidate against the query and return the top_k best, each with its score between 0 and 1.
def rerank(query: str, hits: list, top_k: int, cfg) -> list:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError(f"top_k must be a positive integer; got {top_k!r}")
    if not query.strip() or not hits:
        return []
    model_name = cfg.retrieval.get("reranker_model")
    if not model_name:
        raise ValueError("retrieval.reranker_model must be configured when reranking is enabled")
    # One (question, chunk) pair per candidate. The chunk side is embed_text (course name + position + text),
    # the same text the searches used.
    pairs = [
        (query, hit.chunk.embed_text)
        for hit in hits
    ]
    try:
        scores = _model(model_name).predict(pairs, show_progress_bar=False)
    except Exception as exc:
        raise RuntimeError(f"Could not rerank hits with local model {model_name!r}") from exc
    if len(scores) != len(hits):
        raise ValueError("reranker returned a different number of scores than input hits")
    # The cross-encoder gives raw scores of any size. The sigmoid turns each one into a confidence between
    # 0 and 1, so one fixed threshold (the self-check's 0.2) can be used for every question.
    normalized_scores = []
    for score in scores:
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("reranker returned a non-finite score")
        # Two equivalent forms of the sigmoid, chosen by the sign so exp() never overflows.
        if score >= 0:
            confidence = 1.0 / (1.0 + math.exp(-score))
        else:
            exp_score = math.exp(score)
            confidence = exp_score / (1.0 + exp_score)
        normalized_scores.append(confidence)
    # Highest confidence first; equal scores are ordered by id, so the order is always the same.
    ranked = sorted(
        zip(normalized_scores, hits),
        key=lambda item: (-item[0], item[1].chunk.chunk_id),
    )
    # Keep the best top_k, numbered from 1, with their reranker score.
    return [
        replace(hit, rank=rank, score_rerank=score)
        for rank, (score, hit) in enumerate(ranked[:top_k], start=1)
    ]


# Load the cross-encoder when the app starts (called by src/warmup.py), so the first answer is not slower.
def warm_up(cfg) -> None:
    # Only needed when reranking is switched on (retrieval.rerank).
    if not cfg.retrieval.get("rerank", False):
        return
    model_name = cfg.retrieval.get("reranker_model")
    if not model_name:
        raise ValueError("retrieval.reranker_model must be configured when reranking is enabled")
    _model(model_name)
