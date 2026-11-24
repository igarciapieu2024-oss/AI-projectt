# Keep the knowledge-base files in memory between questions: the chunks file, the BM25 index and the
# ChromaDB connection. Reading them from disk for every question would add seconds to each answer. dense.py
# and bm25.py use it, and so does src/warmup.py, which loads them when the app starts.
from __future__ import annotations

# lru_cache remembers what a function returned, so each file is read only once.
from functools import lru_cache
from pathlib import Path

# The format of a chunk, and the function that reads a JSONL file into chunks.
from src.schemas import Chunk
from src.utils.io import read_jsonl


# The key that identifies one version of a file: its path, its modification time and its size. If the file
# is rebuilt, the key changes, so we load the new version instead of the old copy in memory.
def _file_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


# Read the chunks file once per version and index the chunks by id. lru_cache keeps up to 4 versions, for
# example the knowledge bases of different configurations.
@lru_cache(maxsize=4)
def _load_chunks(path: str, modified_ns: int, size: int) -> dict[str, Chunk]:
    # The time and the size are only part of the cache key; we don't need them inside.
    del modified_ns, size
    chunks = read_jsonl(path, Chunk)
    # A dictionary from chunk id to chunk, so each chunk can be found straight away.
    indexed = {chunk.chunk_id: chunk for chunk in chunks}
    # Two chunks with the same id would make one of them impossible to find.
    if len(indexed) != len(chunks):
        raise ValueError(f"Chunk file contains duplicate chunk IDs: {path}")
    return indexed


# chunk_id -> Chunk, with every field (text, covers...) that ChromaDB doesn't store.
def load_chunks(path: Path) -> dict[str, Chunk]:
    return _load_chunks(*_file_key(path))


# Read the BM25 index (the pickle file written by build_indexes.py) once per version.
@lru_cache(maxsize=4)
def _load_bm25(path: str, modified_ns: int, size: int) -> dict:
    # As above, the time and the size are only part of the cache key.
    del modified_ns, size
    import pickle

    # Open the file in binary mode and load the saved Python object.
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


# The BM25 index of a file, read from memory if this version was already loaded.
def load_bm25(path: Path) -> dict:
    return _load_bm25(*_file_key(path))


# One connection to the ChromaDB folder, opened only once.
@lru_cache(maxsize=2)
def _chroma_client(path: str):
    # Imported here, so the library is only loaded when it is needed.
    import chromadb

    return chromadb.PersistentClient(path=path)


# Return the ChromaDB collection of a knowledge base. A missing collection gives a clear FileNotFoundError,
# so dense.py can tell the user to run the build first.
def get_chroma_collection(path: Path, name: str):
    client = _chroma_client(str(path.resolve()))
    try:
        return client.get_collection(name=name)
    except Exception as exc:
        # Different ChromaDB versions list the collections as objects or as names, so we handle both.
        names = {
            item.name if hasattr(item, "name") else str(item)
            for item in client.list_collections()
        }
        # The collection really doesn't exist: we say so clearly.
        if name not in names:
            raise FileNotFoundError(f"Chroma collection {name!r} is missing") from exc
        # It exists but failed for another reason, so we raise the original error.
        raise


# Empty all the caches, so the next question reads the files again; this is needed after rebuilding the
# indexes in the same Python session.
def clear_caches() -> None:
    _load_chunks.cache_clear()
    _load_bm25.cache_clear()
    _chroma_client.cache_clear()
