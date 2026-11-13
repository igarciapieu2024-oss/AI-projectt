"""Warm local retrieval models before the first question."""
from __future__ import annotations

from src.config import PATHS, bm25_path, chunks_path
from src.generation.llm import chat
from src.retrieval import embed, index_cache, rerank, router


def warm_up(cfg) -> None:
    """Load the models enabled by this configuration into their process caches."""
    model_name = cfg.kb.get("embed_model")
    if not model_name:
        raise ValueError("kb.embed_model must be configured")
    index_cache.get_chroma_collection(PATHS["chroma"], cfg.collection_name)
    index_cache.load_bm25(bm25_path(cfg.kb_sig))
    index_cache.load_chunks(chunks_path(cfg.kb_sig))
    embed.warm_up(model_name)
    rerank.warm_up(cfg)
    router.warm_up(cfg)
    chat(
        [{"role": "user", "content": "Hi"}],
        cfg,
        num_predict=1,
    )
