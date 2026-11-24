# Dense search, which means searching by meaning, in ChromaDB. _build_hits() in src/generation/answer.py
# calls it to get the 20 closest chunks (k_dense in base.yaml) and then merges them with the BM25 results in
# fusion.py. The question is embedded with the same model as the chunks, so the closest vectors belong to the
# chunks whose meaning is closest to the question, even when they use other words.
from __future__ import annotations

# Where ChromaDB and the chunks file of each knowledge base are saved.
from src.config import PATHS, chunks_path
# embed() turns the question into a vector with the embedding model (embed.py).
from src.retrieval.embed import embed
# The ChromaDB collection and the chunks stay in memory between questions (index_cache.py).
from src.retrieval.index_cache import get_chroma_collection, load_chunks
# The shared formats: the search filters and a search result (Hit).
from src.schemas import Filters, Hit


# k is how many results we ask for, so it has to be a whole number of at least 1. True and False are
# rejected too, because Python treats them as integers.
def _validate_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive integer; got {k!r}")


# True if the chunk passes the filters: the right course (if one is set) and at least one of the requested
# sessions (if any are set). With no filters, every chunk passes.
def _matches_filters(chunk, filters: Filters | None) -> bool:
    return filters is None or (
        (filters.course_code is None or chunk.course_code == filters.course_code)
        and (not filters.sessions or any(session in chunk.sessions for session in filters.sessions))
    )


# Return the k chunks closest to the query, as Hits with the whole chunk and its cosine similarity.
def search(query: str, cfg, filters: Filters | None, k: int) -> list[Hit]:
    # An empty query can't be searched.
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    _validate_k(k)

    # The ChromaDB collection of this knowledge base (kb_<kb_sig>); if it doesn't exist, the error tells you
    # to run the build.
    try:
        collection = get_chroma_collection(PATHS["chroma"], cfg.collection_name)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Chroma collection {cfg.collection_name!r} is missing. "
            f"Build indexes with `python -m src.pipeline build --config configs/{cfg.name}.yaml`."
        ) from exc
    # How many vectors the collection has; an empty collection gives no results.
    count = collection.count()
    if not count:
        return []

    # We embed the question with the same model as the chunks (kb.embed_model), so both are in the same space.
    model_name = cfg.kb.get("embed_model")
    if not model_name:
        raise ValueError("kb.embed_model must be set for dense retrieval")
    # embed() gives one vector per text; we take the only one and turn it into a plain list for ChromaDB.
    query_vector = embed([query.strip()], model_name)[0].tolist()
    # The course filter is applied inside ChromaDB (where), using the course_code stored with each vector.
    where = {"course_code": filters.course_code} if filters and filters.course_code else None
    # The sessions are stored as text in ChromaDB, so we can't filter them there: with a session filter we ask
    # for every chunk and check the sessions below; otherwise we only ask for the k closest.
    candidate_count = count if filters and filters.sessions else min(k, count)
    # Ask ChromaDB for the closest vectors, with their metadata and their distances.
    response = collection.query(
        query_embeddings=[query_vector],
        n_results=candidate_count,
        where=where,
        include=["metadatas", "distances"],
    )

    # ChromaDB gives back ids and distances; the full chunks (text, covers...) come from the chunks file.
    chunk_file = chunks_path(cfg.kb_sig)
    if not chunk_file.is_file():
        raise FileNotFoundError(f"Chunk file not found for kb_sig={cfg.kb_sig}: {chunk_file}")
    chunks = load_chunks(chunk_file)
    hits = []
    # ChromaDB answers with one list per question vector; we sent one, so we read [0]. The results go from
    # the closest to the furthest, so the first ones that pass the filters are the best.
    for chunk_id, distance in zip(response["ids"][0], response["distances"][0]):
        chunk = chunks.get(chunk_id)
        # A vector without its chunk means ChromaDB and the chunks file come from different builds.
        if chunk is None:
            raise KeyError(f"Chroma chunk {chunk_id!r} is missing from {chunk_file}")
        if _matches_filters(chunk, filters):
            hits.append(
                Hit(
                    chunk=chunk,
                    # Its position in our list, counting from 1.
                    rank=len(hits) + 1,
                    # ChromaDB gives the cosine distance; 1 - distance is the cosine similarity.
                    score_dense=1.0 - float(distance),
                )
            )
            # We stop once k chunks have passed the filters.
            if len(hits) == k:
                break
    return hits
