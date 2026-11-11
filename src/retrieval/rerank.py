"""Cross-encoder reranker on the fused top hits.

Owner: Sergio.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache


@lru_cache(maxsize=3)
def _model(model_name: str):
    """Load a cached local cross-encoder without attempting a model download."""
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


def rerank(query: str, hits: list, top_k: int, cfg) -> list:
    """Return the highest-scoring top_k hits according to the configured cross-encoder."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError(f"top_k must be a positive integer; got {top_k!r}")
    if not query.strip() or not hits:
        return []
    model_name = cfg.retrieval.get("reranker_model")
    if not model_name:
        raise ValueError("retrieval.reranker_model must be configured when reranking is enabled")
    pairs = [
        (query, f"{hit.chunk.outline_path}\n{hit.chunk.text}")
        for hit in hits
    ]
    try:
        scores = _model(model_name).predict(pairs, show_progress_bar=False)
    except Exception as exc:
        raise RuntimeError(f"Could not rerank hits with local model {model_name!r}") from exc
    if len(scores) != len(hits):
        raise ValueError("reranker returned a different number of scores than input hits")
    ranked = sorted(
        (
            (float(score), hit)
            for hit, score in zip(hits, scores)
        ),
        key=lambda item: (-item[0], item[1].chunk.chunk_id),
    )
    return [
        replace(hit, rank=rank, score_rerank=score)
        for rank, (score, hit) in enumerate(ranked[:top_k], start=1)
    ]
