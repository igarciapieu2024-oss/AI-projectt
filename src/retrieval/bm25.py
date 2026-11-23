# Keyword search with BM25. Called by _build_hits() in src/generation/answer.py, which asks for the 20
# best chunks (k_bm25 in base.yaml) and merges them with the dense results in fusion.py. BM25 finds exact
# words, names and numbers that the vectors of the dense search can miss. The question is split into
# words with tokenize() from build_indexes.py, the same function used on the chunks.
from __future__ import annotations

from src.config import bm25_path, chunks_path
from src.kb.build_indexes import tokenize
from src.retrieval.index_cache import load_bm25, load_chunks
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


# Return the k chunks with the highest BM25 score for the query, as Hits with the complete chunk and its score.
def search(query: str, cfg, filters: Filters | None, k: int) -> list[Hit]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    _validate_k(k)
    # A question with no words (only symbols) cannot match anything.
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    # The BM25 index of this knowledge base; if it does not exist, the error says to run the build.
    index_path = bm25_path(cfg.kb_sig)
    if not index_path.is_file():
        raise FileNotFoundError(
            f"BM25 index not found: {index_path}. "
            f"Build indexes with `python -m src.pipeline build --config configs/{cfg.name}.yaml`."
        )
    index = load_bm25(index_path)
    # Check the file has the four parts build_indexes.py writes, and that it belongs to this configuration.
    expected_keys = {"kb_sig", "chunk_ids", "tokenized_corpus", "bm25"}
    if not isinstance(index, dict) or set(index) != expected_keys:
        raise ValueError(f"BM25 index at {index_path} has an invalid format")
    if index["kb_sig"] != cfg.kb_sig:
        raise ValueError(
            f"BM25 index kb_sig mismatch: expected {cfg.kb_sig}, found {index['kb_sig']}"
        )
    # The chunk ids are in the same order as the scores BM25 returns, so both lists must have the same length.
    chunk_ids = index["chunk_ids"]
    tokenized_corpus = index["tokenized_corpus"]
    if not isinstance(chunk_ids, list) or not isinstance(tokenized_corpus, list):
        raise ValueError(f"BM25 index at {index_path} has invalid corpus data")
    if len(chunk_ids) != len(tokenized_corpus):
        raise ValueError(f"BM25 index at {index_path} has mismatched corpus lengths")

    # The complete chunks (text, covers...) come from the chunks file.
    chunk_file = chunks_path(cfg.kb_sig)
    if not chunk_file.is_file():
        raise FileNotFoundError(f"Chunk file not found for kb_sig={cfg.kb_sig}: {chunk_file}")
    chunks = load_chunks(chunk_file)
    # One score per chunk: how many words of the question it contains, weighted by how rare each word is.
    try:
        scores = index["bm25"].get_scores(query_tokens)
    except (AttributeError, TypeError) as exc:
        raise ValueError(f"BM25 index at {index_path} has no compatible scorer") from exc
    if len(scores) != len(chunk_ids):
        raise ValueError(f"BM25 scorer at {index_path} returned an invalid score count")

    scored = []
    for chunk_id, score in zip(chunk_ids, scores):
        chunk = chunks.get(chunk_id)
        # A scored id without its chunk means the BM25 index and the chunks file come from different builds.
        if chunk is None:
            raise KeyError(f"BM25 chunk {chunk_id!r} is missing from {chunk_file}")
        # Keep only chunks that share at least one word with the question (score above 0), pass the
        # filters and have text.
        if score > 0 and _matches_filters(chunk, filters) and chunk.text.strip():
            scored.append((float(score), chunk))
    # Highest score first; equal scores are ordered by id, so the order is always the same.
    scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
    return [
        Hit(chunk=chunk, rank=rank, score_bm25=score)
        for rank, (score, chunk) in enumerate(scored[:k], start=1)
    ]
