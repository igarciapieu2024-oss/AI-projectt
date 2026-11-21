# Keep the knowledge-base files in memory between questions: the chunks file, the BM25 index and the
# ChromaDB connection. Reading them from disk for every question would add seconds to each answer.
# Used by dense.py and bm25.py, and by src/warmup.py, which loads them when the app starts.
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.schemas import Chunk
from src.utils.io import read_jsonl


# The key that identifies one version of a file: its path, its modification time and its size.
# If the file is rebuilt, the key changes and the new version is loaded instead of the old copy in memory.
def _file_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


# Read the chunks file once per version and index the chunks by id. lru_cache keeps up to 4 versions,
# for example the knowledge bases of different configurations.
@lru_cache(maxsize=4)
def _load_chunks(path: str, modified_ns: int, size: int) -> dict[str, Chunk]:
    # The time and size are only part of the cache key; they are not needed inside.
    del modified_ns, size
    chunks = read_jsonl(path, Chunk)
    indexed = {chunk.chunk_id: chunk for chunk in chunks}
    # Two chunks with the same id would make one of them impossible to find.
    if len(indexed) != len(chunks):
        raise ValueError(f"Chunk file contains duplicate chunk IDs: {path}")
    return indexed


# chunk_id -> Chunk, with every field (text, covers...) that ChromaDB does not store.
def load_chunks(path: Path) -> dict[str, Chunk]:
    return _load_chunks(*_file_key(path))


# Read the BM25 index (the pickle written by build_indexes.py) once per version.
@lru_cache(maxsize=4)
def _load_bm25(path: str, modified_ns: int, size: int) -> dict:
    del modified_ns, size
    import pickle

    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def load_bm25(path: Path) -> dict:
    return _load_bm25(*_file_key(path))


# One connection to the ChromaDB folder, opened once.
@lru_cache(maxsize=2)
def _chroma_client(path: str):
    import chromadb

    return chromadb.PersistentClient(path=path)


# Return the ChromaDB collection of a knowledge base. A missing collection gives a clear
# FileNotFoundError, so dense.py can tell the user to run the build first.
def get_chroma_collection(path: Path, name: str):
    client = _chroma_client(str(path.resolve()))
    try:
        return client.get_collection(name=name)
    except Exception as exc:
        # Different ChromaDB versions list collections as objects or as names.
        names = {
            item.name if hasattr(item, "name") else str(item)
            for item in client.list_collections()
        }
        if name not in names:
            raise FileNotFoundError(f"Chroma collection {name!r} is missing") from exc
        raise


# Empty all the caches, so the next question reads the files again; needed after rebuilding the
# indexes in the same Python session.
def clear_caches() -> None:
    _load_chunks.cache_clear()
    _load_bm25.cache_clear()
    _chroma_client.cache_clear()
