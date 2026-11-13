"""Load Francesca's trained router and fall back to the local LLM when unsure.

Owner: Sergio (wrapper), Francesca (model).
"""
from __future__ import annotations

import logging
import math
from functools import lru_cache

from src.config import PATHS
from src.generation.llm import chat
from src.schemas import RouteDecision

logger = logging.getLogger(__name__)
_LABELS = {"course_admin", "course_content", "cross_course", "out_of_scope"}


def _none() -> RouteDecision:
    return RouteDecision(label="none", prob=0.0, source="none")


@lru_cache(maxsize=1)
def _load_router(model_path: str):
    from src.train.train_router import load_router

    return load_router(model_path)


def _route_with_llm(question: str, cfg) -> RouteDecision:
    from src.train.train_router import LLM_PROMPT, parse_llm_label

    response = chat(
        [
            {"role": "system", "content": LLM_PROMPT},
            {"role": "user", "content": question},
        ],
        cfg,
    )
    if not isinstance(response, str):
        raise TypeError("chat() must return text when stream=False")
    label = parse_llm_label(response)
    if label not in _LABELS:
        logger.warning("LLM router returned an invalid label: %r", response)
        return _none()
    return RouteDecision(label=label, prob=1.0, source="llm")


def route(question: str, cfg) -> RouteDecision:
    """Route a question using the configured classifier or local LLM."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be empty")

    routing = getattr(cfg, "routing", {})
    mode = routing.get("router", "none")
    if mode == "none":
        return _none()
    if mode not in {"classifier", "llm"}:
        logger.error("Unsupported routing.router mode: %r", mode)
        return _none()
    if mode == "llm":
        try:
            return _route_with_llm(question.strip(), cfg)
        except Exception:
            logger.exception("LLM routing failed; continuing without a route")
            return _none()

    model_path = PATHS["router_model"]
    if not model_path.is_file():
        logger.warning("Router model is missing: %s", model_path)
        return _none()
    try:
        model = _load_router(str(model_path.resolve()))
        label, probability = model.route(question.strip())
        probability = float(probability)
        if label not in _LABELS or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"router returned an invalid prediction: {label!r}, {probability!r}")
        threshold = routing.get("min_prob", 0.6)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(f"routing.min_prob must be numeric; got {threshold!r}")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"routing.min_prob must be between 0 and 1; got {threshold!r}")
    except Exception:
        logger.exception("Classifier routing failed; continuing without a route")
        return _none()

    if probability < threshold:
        try:
            return _route_with_llm(question.strip(), cfg)
        except Exception:
            logger.exception("LLM routing fallback failed; continuing without a route")
            return _none()
    return RouteDecision(label=label, prob=probability, source="classifier")


def clear_router_cache() -> None:
    """Clear the cached classifier after replacing its model artifact."""
    _load_router.cache_clear()


def warm_up(cfg) -> None:
    """Load the configured classifier and its embedder before the first question."""
    if getattr(cfg, "routing", {}).get("router", "none") != "classifier":
        return
    model_path = PATHS["router_model"]
    if not model_path.is_file():
        logger.warning("Router model is missing: %s", model_path)
        return
    try:
        _load_router(str(model_path.resolve()))
    except Exception:
        logger.exception("Classifier warm-up failed; routing will continue without classifier")
