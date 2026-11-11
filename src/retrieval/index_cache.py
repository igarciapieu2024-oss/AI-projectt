"""Cache local knowledge-base artifacts between retrieval calls."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.schemas import Chunk
from src.utils.io import read_jsonl


def _file_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=4)
def _load_chunks(path: str, modified_ns: int, size: int) -> dict[str, Chunk]:
    del modified_ns, size
    chunks = read_jsonl(path, Chunk)
    indexed = {chunk.chunk_id: chunk for chunk in chunks}
    if len(indexed) != len(chunks):
        raise ValueError(f"Chunk file contains duplicate chunk IDs: {path}")
    return indexed


def load_chunks(path: Path) -> dict[str, Chunk]:
    """Load complete chunk records once per file version."""
    return _load_chunks(*_file_key(path))


@lru_cache(maxsize=4)
def _load_bm25(path: str, modified_ns: int, size: int) -> dict:
    del modified_ns, size
    import pickle

    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def load_bm25(path: Path) -> dict:
    """Load the serialized BM25 index once per file version."""
    return _load_bm25(*_file_key(path))


@lru_cache(maxsize=2)
def _chroma_client(path: str):
    import chromadb

    return chromadb.PersistentClient(path=path)


def get_chroma_collection(path: Path, name: str):
    """Return a cached Chroma collection and report a missing index clearly."""
    client = _chroma_client(str(path.resolve()))
    try:
        return client.get_collection(name=name)
    except Exception as exc:
        names = {
            item.name if hasattr(item, "name") else str(item)
            for item in client.list_collections()
        }
        if name not in names:
            raise FileNotFoundError(f"Chroma collection {name!r} is missing") from exc
        raise


def clear_caches() -> None:
    """Clear cached index artifacts, e.g. after rebuilding indexes in this process."""
    _load_chunks.cache_clear()
    _load_bm25.cache_clear()
    _chroma_client.cache_clear()
