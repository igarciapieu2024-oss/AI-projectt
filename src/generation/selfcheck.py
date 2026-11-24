# Self-check: refuse to answer when the retrieved material is too weak. Called by _prepare_answer() in
# src/generation/answer.py after the reranker. If even the best source scores below the threshold, no
# prompt is built and the student gets the refusal message instead of an answer the LLM would have to guess.
from __future__ import annotations

# math.isfinite() checks that the threshold is a real number.
import math


# True when the sources are too weak to answer: there are none, or the best reranker confidence is below
# generation.selfcheck_threshold (0.2 in base.yaml, tuned on the dev questions).
def is_weak(hits: list, cfg) -> bool:
    # No sources at all: there is nothing to answer from.
    if not hits:
        return True
    # The threshold from the config (generation.selfcheck_threshold); 0.2 if it isn't set.
    threshold = cfg.generation.get("selfcheck_threshold", 0.2)
    # The threshold must be a real number (true/false is rejected, although Python counts it as a number).
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(f"generation.selfcheck_threshold must be numeric; got {threshold!r}")
    # Infinity or NaN can't be used as a threshold.
    if not math.isfinite(threshold):
        raise ValueError("generation.selfcheck_threshold must be finite")
    # The reranker confidences (0 to 1, from rerank.py); without reranking there is nothing to compare.
    scores = [hit.score_rerank for hit in hits]
    # A source without a reranker score means the reranker didn't run, so we can't decide.
    if any(score is None for score in scores):
        raise ValueError("self-check requires reranked hits with score_rerank values")
    # Only the best source matters: one good source is enough to answer.
    return max(scores) < threshold
