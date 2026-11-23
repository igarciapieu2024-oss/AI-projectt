# Dense search (search by meaning) in ChromaDB. Called by _build_hits() in src/generation/answer.py,
# which asks for the 20 closest chunks (k_dense in base.yaml) and merges them with the BM25 results in
# fusion.py. The question is embedded with the same model as the chunks, so the closest vectors are the
# chunks whose meaning is closest to the question, even when they use other words.
from __future__ import annotations

from src.config import PATHS, chunks_path
from src.retrieval.embed import embed
from src.retrieval.index_cache import get_chroma_collection, load_chunks
from src.schemas import Filters, Hit


# k is the number of results asked for; it must be a whole number of at least 1.
def _validate_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive integer; got {k!r}")


# True if the chunk passes the filters: the right course (if one is set) and at least one of the
# requested sessions (if any are set). No filters means every chunk passes.
def _matches_filters(chunk, filters: Filters | None) -> bool:
    return filters is None or (
        (filters.course_code is None or chunk.course_code == filters.course_code)
        and (not filters.sessions or any(session in chunk.sessions for session in filters.sessions))
    )


# Return the k chunks closest to the query, as Hits with the complete chunk and its cosine similarity.
def search(query: str, cfg, filters: Filters | None, k: int) -> list[Hit]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    _validate_k(k)

    # The collection of this knowledge base (kb_<kb_sig>); if it does not exist, the error says to run the build.
    try:
        collection = get_chroma_collection(PATHS["chroma"], cfg.collection_name)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Chroma collection {cfg.collection_name!r} is missing. "
            f"Build indexes with `python -m src.pipeline build --config configs/{cfg.name}.yaml`."
        ) from exc
    count = collection.count()
    if not count:
        return []

    # Embed the question with the same model as the chunks.
    model_name = cfg.kb.get("embed_model")
    if not model_name:
        raise ValueError("kb.embed_model must be set for dense retrieval")
    query_vector = embed([query.strip()], model_name)[0].tolist()
    # The course filter is applied inside ChromaDB (where), using the course_code stored with each vector.
    where = {"course_code": filters.course_code} if filters and filters.course_code else None
    # The sessions are stored as text in ChromaDB, so they cannot be filtered there: with a session filter
    # every chunk is requested and the sessions are checked below; otherwise only the k closest are requested.
    candidate_count = count if filters and filters.sessions else min(k, count)
    response = collection.query(
        query_embeddings=[query_vector],
        n_results=candidate_count,
        where=where,
        include=["metadatas", "distances"],
    )

    # ChromaDB returns ids and distances; the complete chunks (text, covers...) come from the chunks file.
    chunk_file = chunks_path(cfg.kb_sig)
    if not chunk_file.is_file():
        raise FileNotFoundError(f"Chunk file not found for kb_sig={cfg.kb_sig}: {chunk_file}")
    chunks = load_chunks(chunk_file)
    hits = []
    for chunk_id, distance in zip(response["ids"][0], response["distances"][0]):
        chunk = chunks.get(chunk_id)
        # A vector without its chunk means ChromaDB and the chunks file come from different builds.
        if chunk is None:
            raise KeyError(f"Chroma chunk {chunk_id!r} is missing from {chunk_file}")
        if _matches_filters(chunk, filters):
            hits.append(
                Hit(
                    chunk=chunk,
                    rank=len(hits) + 1,
                    # ChromaDB gives the cosine distance; 1 - distance is the cosine similarity.
                    score_dense=1.0 - float(distance),
                )
            )
            # Stop once k chunks have passed the filters.
            if len(hits) == k:
                break
    return hits
