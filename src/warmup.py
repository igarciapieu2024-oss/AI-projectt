# Load every model and index into memory before the first question, so the first answer is not slower than
# the rest. Called by app_components/engine.py when the app starts.
from __future__ import annotations

# Where the indexes are, the function that talks to the LLM, and the modules whose models we load.
from src.config import PATHS, bm25_path, chunks_path
from src.generation.llm import chat
from src.retrieval import embed, index_cache, rerank, router


# Load what this configuration uses: the ChromaDB collection, the BM25 index, the chunks, the embedding
# model, the reranker and the router.
def warm_up(cfg) -> None:
    # The embedding model has to be set in the config (kb.embed_model).
    model_name = cfg.kb.get("embed_model")
    if not model_name:
        raise ValueError("kb.embed_model must be configured")
    # The three knowledge-base files: once read, index_cache.py keeps them in memory.
    index_cache.get_chroma_collection(PATHS["chroma"], cfg.collection_name)
    index_cache.load_bm25(bm25_path(cfg.kb_sig))
    index_cache.load_chunks(chunks_path(cfg.kb_sig))
    # The three models of the search: the embedding model, the reranker and the router.
    embed.warm_up(model_name)
    rerank.warm_up(cfg)
    router.warm_up(cfg)
    # One very short call to the LLM (one token), so Ollama loads the model into memory now.
    chat(
        [{"role": "user", "content": "Hi"}],
        cfg,
        num_predict=1,
    )
