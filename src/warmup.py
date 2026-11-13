"""Warm local retrieval models before the first question."""
from __future__ import annotations

from src.retrieval import embed, rerank, router


def warm_up(cfg) -> None:
    """Load the models enabled by this configuration into their process caches."""
    model_name = cfg.kb.get("embed_model")
    if not model_name:
        raise ValueError("kb.embed_model must be configured")
    embed.warm_up(model_name)
    rerank.warm_up(cfg)
    router.warm_up(cfg)
