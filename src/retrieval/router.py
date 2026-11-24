# Decide the path of a question with the trained classifier (the SVM saved in models/router.joblib by
# src/train/train_router.py), and ask the local LLM instead when the classifier isn't sure.
# route() is called by _prepare_answer() in src/generation/answer.py with the question prepared in step 1.
# Its label decides the path: course_admin and course_content go to the normal search, cross_course is
# answered from the course table, out_of_scope is refused without searching, and "none" (no label) is
# searched normally.
from __future__ import annotations

# logging writes warnings and errors without stopping the answer.
import logging
# math.isfinite() checks that a probability is a real number.
import math
# lru_cache keeps the loaded classifier in memory, so it is only loaded the first time.
from functools import lru_cache

# Where the saved classifier is.
from src.config import PATHS
# chat() sends messages to the local LLM, which decides when the classifier isn't sure.
from src.generation.llm import chat
# The format of the router's decision: label, probability and who decided.
from src.schemas import RouteDecision

logger = logging.getLogger(__name__)
# The four labels the router can give; anything else is treated as an invalid answer.
_LABELS = {"course_admin", "course_content", "cross_course", "out_of_scope"}


# The result we use when no label can be given (router switched off, model missing or an error). The
# question is then searched normally instead of stopping the answer.
def _none() -> RouteDecision:
    return RouteDecision(label="none", prob=0.0, source="none")


# Load the saved classifier and its embedding model once and keep them in memory (lru_cache), so we don't
# load them again for every question.
@lru_cache(maxsize=1)
def _load_router(model_path: str):
    # Imported here, so the training code is only loaded when the router is used.
    from src.train.train_router import load_router

    return load_router(model_path)


# Ask the LLM for the label. The instruction (LLM_PROMPT) and the way we read its answer (parse_llm_label)
# come from train_router.py: they are the same ones we used when we compared the LLM with the classifiers,
# so the LLM router works here just as it did in that comparison.
def _route_with_llm(question: str, cfg) -> RouteDecision:
    from src.train.train_router import LLM_PROMPT, parse_llm_label

    # The instruction as the system message, and the question as the user message.
    response = chat(
        [
            {"role": "system", "content": LLM_PROMPT},
            {"role": "user", "content": question},
        ],
        cfg,
    )
    # Without streaming, chat() has to give back text.
    if not isinstance(response, str):
        raise TypeError("chat() must return text when stream=False")
    # We keep only the label from the model's answer.
    label = parse_llm_label(response)
    # An answer that isn't one of the four labels gives no label, instead of a wrong one.
    if label not in _LABELS:
        logger.warning("LLM router returned an invalid label: %r", response)
        return _none()
    # The LLM gives no probability, so 1.0 marks "decided by the LLM".
    return RouteDecision(label=label, prob=1.0, source="llm")


# Return the label of the question, its probability and who decided it (classifier, llm or none). Every
# error is caught and gives "none", so a problem with the router never stops the answer.
def route(question: str, cfg) -> RouteDecision:
    # An empty question can't be routed.
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be empty")

    # routing.router in the config chooses who classifies: none (no router), classifier (the SVM, with the
    # LLM when it isn't sure) or llm (always the LLM). final_system.yaml uses classifier.
    routing = getattr(cfg, "routing", {})
    mode = routing.get("router", "none")
    # No router: every question gets no label and is searched normally.
    if mode == "none":
        return _none()
    # Any other value is a mistake in the config; we log it and carry on without a label.
    if mode not in {"classifier", "llm"}:
        logger.error("Unsupported routing.router mode: %r", mode)
        return _none()
    # LLM mode: the LLM decides every question.
    if mode == "llm":
        try:
            return _route_with_llm(question.strip(), cfg)
        except Exception:
            logger.exception("LLM routing failed; continuing without a route")
            return _none()

    # Classifier mode: the model file has to exist (it is in the repository, made by train_router.py).
    model_path = PATHS["router_model"]
    if not model_path.is_file():
        logger.warning("Router model is missing: %s", model_path)
        return _none()
    try:
        model = _load_router(str(model_path.resolve()))
        # The classifier embeds the question and returns its most likely label and that label's probability.
        label, probability = model.route(question.strip())
        probability = float(probability)
        # We check that the prediction is one of the four labels and a probability between 0 and 1.
        if label not in _LABELS or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"router returned an invalid prediction: {label!r}, {probability!r}")
        # min_prob (0.6 in base.yaml): below it, the classifier isn't sure enough and the LLM decides.
        threshold = routing.get("min_prob", 0.6)
        # The threshold has to be a number between 0 and 1 (True and False are rejected too).
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(f"routing.min_prob must be numeric; got {threshold!r}")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"routing.min_prob must be between 0 and 1; got {threshold!r}")
    # Any problem with the classifier: we log it and carry on without a label.
    except Exception:
        logger.exception("Classifier routing failed; continuing without a route")
        return _none()

    # The classifier is fast (about 0.01 s per question) and the LLM is slower but more accurate, so we only
    # ask the LLM about the questions the classifier isn't sure of.
    if probability < threshold:
        try:
            return _route_with_llm(question.strip(), cfg)
        except Exception:
            logger.exception("LLM routing fallback failed; continuing without a route")
            return _none()
    # The classifier is sure enough: its label is the decision.
    return RouteDecision(label=label, prob=probability, source="classifier")


# Empty the memory of _load_router(), so a newly trained model file is loaded the next time.
def clear_router_cache() -> None:
    _load_router.cache_clear()


# Load the classifier and its embedding model before the first question, so the first answer isn't slower
# than the rest. src/warmup.py calls it when the app starts.
def warm_up(cfg) -> None:
    # It's only needed when the classifier is used.
    if getattr(cfg, "routing", {}).get("router", "none") != "classifier":
        return
    model_path = PATHS["router_model"]
    if not model_path.is_file():
        logger.warning("Router model is missing: %s", model_path)
        return
    try:
        _load_router(str(model_path.resolve()))
    except Exception:
        # A failed warm-up isn't fatal: route() will try again and fall back to no label.
        logger.exception("Classifier warm-up failed; routing will continue without classifier")
