# Keyword search with BM25. _build_hits() in src/generation/answer.py calls it to get the 20 best chunks
# (k_bm25 in base.yaml) and then merges them with the dense results in fusion.py. We need BM25 because it finds
# exact words, names and numbers that the vectors of the dense search can miss. The question is split into
# words with tokenize() from build_indexes.py, the same function we used on the chunks.
from __future__ import annotations

# Where the BM25 index and the chunks file of each knowledge base are saved.
from src.config import bm25_path, chunks_path
# The same tokenizer we used to build the index, so the question is split into words exactly like the chunks.
from src.kb.build_indexes import tokenize
# The index and the chunks stay in memory between questions (index_cache.py), so we don't read them again.
from src.retrieval.index_cache import load_bm25, load_chunks
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


# Return the k chunks with the highest BM25 score for the query, as Hits with the whole chunk and its score.
def search(query: str, cfg, filters: Filters | None, k: int) -> list[Hit]:
    # An empty query can't be searched.
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    _validate_k(k)
    # Split the question into words. A question with no words (only symbols) can't match anything.
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    # The BM25 index of this knowledge base; if it doesn't exist, the error tells you to run the build.
    index_path = bm25_path(cfg.kb_sig)
    if not index_path.is_file():
        raise FileNotFoundError(
            f"BM25 index not found: {index_path}. "
            f"Build indexes with `python -m src.pipeline build --config configs/{cfg.name}.yaml`."
        )
    index = load_bm25(index_path)
    # We check that the file has the four parts build_indexes.py writes, and that it belongs to this
    # configuration (the same kb_sig).
    expected_keys = {"kb_sig", "chunk_ids", "tokenized_corpus", "bm25"}
    if not isinstance(index, dict) or set(index) != expected_keys:
        raise ValueError(f"BM25 index at {index_path} has an invalid format")
    if index["kb_sig"] != cfg.kb_sig:
        raise ValueError(
            f"BM25 index kb_sig mismatch: expected {cfg.kb_sig}, found {index['kb_sig']}"
        )
    # The chunk ids are in the same order as the scores BM25 gives back, so both lists must have the same length.
    chunk_ids = index["chunk_ids"]
    tokenized_corpus = index["tokenized_corpus"]
    if not isinstance(chunk_ids, list) or not isinstance(tokenized_corpus, list):
        raise ValueError(f"BM25 index at {index_path} has invalid corpus data")
    if len(chunk_ids) != len(tokenized_corpus):
        raise ValueError(f"BM25 index at {index_path} has mismatched corpus lengths")

    # The full chunks (text, covers...) come from the chunks file, because the index only keeps their words.
    chunk_file = chunks_path(cfg.kb_sig)
    if not chunk_file.is_file():
        raise FileNotFoundError(f"Chunk file not found for kb_sig={cfg.kb_sig}: {chunk_file}")
    chunks = load_chunks(chunk_file)
    # One score per chunk: it grows with the words of the question the chunk contains, and rare words count
    # more than common ones.
    try:
        scores = index["bm25"].get_scores(query_tokens)
    # If the saved object can't give scores, the index file is broken or comes from an incompatible version.
    except (AttributeError, TypeError) as exc:
        raise ValueError(f"BM25 index at {index_path} has no compatible scorer") from exc
    # There has to be exactly one score per chunk.
    if len(scores) != len(chunk_ids):
        raise ValueError(f"BM25 scorer at {index_path} returned an invalid score count")

    scored = []
    for chunk_id, score in zip(chunk_ids, scores):
        chunk = chunks.get(chunk_id)
        # A scored id without its chunk means the BM25 index and the chunks file come from different builds.
        if chunk is None:
            raise KeyError(f"BM25 chunk {chunk_id!r} is missing from {chunk_file}")
        # We only keep the chunks that share at least one word with the question (score above 0), pass the
        # filters and have some text.
        if score > 0 and _matches_filters(chunk, filters) and chunk.text.strip():
            scored.append((float(score), chunk))
    # Highest score first; equal scores are ordered by id, so the order is always the same.
    scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
    # The best k, numbered from 1, with their BM25 score (the app shows it in its "Behind the scenes" panel).
    return [
        Hit(chunk=chunk, rank=rank, score_bm25=score)
        for rank, (score, chunk) in enumerate(scored[:k], start=1)
    ]
