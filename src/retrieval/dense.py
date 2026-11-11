"""Meaning search in ChromaDB.

Owner: Sergio.
"""
from __future__ import annotations

from src.config import PATHS, chunks_path
from src.retrieval.embed import embed
from src.schemas import Chunk, Filters, Hit
from src.utils.io import read_jsonl


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive integer; got {k!r}")


def _matches_filters(chunk, filters: Filters | None) -> bool:
    return filters is None or (
        (filters.course_code is None or chunk.course_code == filters.course_code)
        and (not filters.sessions or any(session in chunk.sessions for session in filters.sessions))
    )


def search(query: str, cfg, filters: Filters | None, k: int) -> list[Hit]:
    """Return the k nearest indexed chunks, with complete Chunk records and cosine scores."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    _validate_k(k)

    import chromadb

    client = chromadb.PersistentClient(path=str(PATHS["chroma"]))
    collection_names = {
        item.name if hasattr(item, "name") else str(item)
        for item in client.list_collections()
    }
    if cfg.collection_name not in collection_names:
        raise FileNotFoundError(
            f"Chroma collection {cfg.collection_name!r} is missing. "
            f"Build indexes with `python -m src.pipeline build --config configs/{cfg.name}.yaml`."
        )
    collection = client.get_collection(name=cfg.collection_name)
    count = collection.count()
    if not count:
        return []

    model_name = cfg.kb.get("embed_model")
    if not model_name:
        raise ValueError("kb.embed_model must be set for dense retrieval")
    query_vector = embed([query.strip()], model_name)[0].tolist()
    where = {"course_code": filters.course_code} if filters and filters.course_code else None
    candidate_count = count if filters and filters.sessions else min(k, count)
    response = collection.query(
        query_embeddings=[query_vector],
        n_results=candidate_count,
        where=where,
        include=["metadatas", "distances"],
    )

    chunk_file = chunks_path(cfg.kb_sig)
    if not chunk_file.is_file():
        raise FileNotFoundError(f"Chunk file not found for kb_sig={cfg.kb_sig}: {chunk_file}")
    chunks = {chunk.chunk_id: chunk for chunk in read_jsonl(chunk_file, Chunk)}
    hits = []
    for chunk_id, distance in zip(response["ids"][0], response["distances"][0]):
        chunk = chunks.get(chunk_id)
        if chunk is None:
            raise KeyError(f"Chroma chunk {chunk_id!r} is missing from {chunk_file}")
        if _matches_filters(chunk, filters):
            hits.append(
                Hit(
                    chunk=chunk,
                    rank=len(hits) + 1,
                    score_dense=1.0 - float(distance),
                )
            )
            if len(hits) == k:
                break
    return hits
