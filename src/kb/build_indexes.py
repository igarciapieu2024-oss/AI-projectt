"""Build persistent ChromaDB and BM25 indexes from deduplicated chunks."""
from __future__ import annotations

import logging
import os
import pickle
import re
import tempfile
import uuid
from pathlib import Path

from src.config import PATHS, bm25_path, chunks_path
from src.schemas import Chunk
from src.utils.io import read_jsonl

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[^\W_]+|%", flags=re.UNICODE)
_BM25_KEYS = {"kb_sig", "chunk_ids", "tokenized_corpus", "bm25"}


def tokenize(text: str) -> list[str]:
    """Return lowercase word, number, and percent tokens for BM25."""
    return _TOKEN_PATTERN.findall((text or "").lower())


def _metadata(chunk: Chunk) -> dict:
    """Flatten Chunk fields to scalar Chroma metadata values."""
    return {
        "chunk_id": chunk.chunk_id,
        "kb_sig": chunk.kb_sig,
        "strategy": chunk.strategy,
        "doc_type": chunk.doc_type,
        "course_code": chunk.course_code,
        "source_id": chunk.source_id,
        # Chroma metadata values are scalar; semicolons preserve the session list.
        "sessions": ";".join(str(value) for value in chunk.sessions),
        "parent_id": chunk.parent_id,
        "outline_path": chunk.outline_path,
        "has_image_text": chunk.has_image_text,
    }


def load_chunk_by_id(chunk_id: str, cfg) -> Chunk:
    """Reconstruct a complete Chunk by looking up its ID in this KB's JSONL.

    Chroma stores only scalar metadata and ``embed_text``. Callers that need the
    full schema record (including ``text``, ``covers`` and ``duplicate_of``)
    should hydrate a Chroma result using this helper. For multiple results,
    callers should read ``chunks_path(cfg.kb_sig)`` once and cache by chunk_id.
    """
    chunk_file = chunks_path(cfg.kb_sig)
    if not chunk_file.is_file():
        raise FileNotFoundError(f"Chunk file not found for kb_sig={cfg.kb_sig}: {chunk_file}")
    matches = [chunk for chunk in read_jsonl(chunk_file, Chunk) if chunk.chunk_id == chunk_id]
    if not matches:
        raise KeyError(f"Chunk ID {chunk_id!r} not found for kb_sig={cfg.kb_sig}")
    if len(matches) > 1:
        raise ValueError(f"Chunk ID {chunk_id!r} occurs more than once for kb_sig={cfg.kb_sig}")
    return matches[0]


def _embed(texts: list[str], model_name: str):
    """Load the configured model lazily and return normalized vectors."""
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(model_name)
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        logger.exception("Embedding failed while building indexes with model %s", model_name)
        raise RuntimeError(f"Could not embed chunks with model {model_name!r}") from exc
    if len(vectors) != len(texts):
        raise ValueError("Embedding model returned a different number of vectors than input chunks")
    return vectors


def _new_collection_name(base: str, kind: str) -> str:
    """Create a valid, short-lived Chroma collection name (max 63 chars)."""
    suffix = f"__{kind}_{uuid.uuid4().hex[:10]}"
    return f"{base[: 63 - len(suffix)]}{suffix}"


def _collection_names(client) -> set[str]:
    return {item.name if hasattr(item, "name") else str(item) for item in client.list_collections()}


def _delete_collection_if_present(client, name: str) -> None:
    if name in _collection_names(client):
        client.delete_collection(name=name)


def _populate_staging_collection(client, name: str, chunks: list[Chunk], vectors, model_name: str) -> None:
    """Create and validate a complete cosine-space staging collection."""
    collection = client.create_collection(
        name=name,
        metadata={
            "kb_sig": chunks[0].kb_sig,
            "embed_model": model_name,
            "hnsw:space": "cosine",
        },
    )
    collection.add(
        ids=[chunk.chunk_id for chunk in chunks],
        embeddings=vectors.tolist(),
        documents=[chunk.embed_text for chunk in chunks],
        metadatas=[_metadata(chunk) for chunk in chunks],
    )
    stored = collection.get(limit=len(chunks), include=["metadatas"])
    expected_ids = [chunk.chunk_id for chunk in chunks]
    if collection.count() != len(chunks) or set(stored["ids"]) != set(expected_ids):
        raise RuntimeError(f"Staging collection {name!r} failed its count/ID validation")


def _write_bm25_staging(path: Path, payload: dict) -> Path:
    """Write, reload, and validate BM25 in a temporary file beside its target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.stage.", delete=False) as stream:
            temp_path = Path(stream.name)
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        with temp_path.open("rb") as stream:
            check = pickle.load(stream)
        if set(check) != _BM25_KEYS:
            raise ValueError("Staged BM25 pickle has unexpected keys")
        if check["chunk_ids"] != payload["chunk_ids"] or check["tokenized_corpus"] != payload["tokenized_corpus"]:
            raise ValueError("Staged BM25 pickle failed corpus-order validation")
        return temp_path
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        logger.exception("Could not stage BM25 index for %s", path)
        raise


def _commit_staged_indexes(client, final_name: str, stage_name: str, staged_bm25: Path, final_bm25: Path) -> str | None:
    """Promote staged artifacts, rolling Chroma back if the BM25 swap fails.

    Chroma rename and filesystem replacement cannot form one cross-resource
    transaction. Caught failures are rolled back; a process/system crash during
    the short rename window can leave a backup or staging collection for recovery.
    """
    names = _collection_names(client)
    had_previous = final_name in names
    backup_name = _new_collection_name(final_name, "backup") if had_previous else None
    backup_renamed = False
    stage_promoted = False
    try:
        if backup_name:
            client.get_collection(name=final_name).modify(name=backup_name)
            backup_renamed = True
        client.get_collection(name=stage_name).modify(name=final_name)
        stage_promoted = True
        os.replace(staged_bm25, final_bm25)
    except Exception as exc:
        logger.exception("Index promotion failed for %s; attempting rollback", final_name)
        rollback_errors = []
        if stage_promoted:
            try:
                client.get_collection(name=final_name).modify(name=stage_name)
            except Exception as rollback_exc:  # preserve backup for manual recovery
                rollback_errors.append(rollback_exc)
        if backup_renamed and backup_name:
            try:
                client.get_collection(name=backup_name).modify(name=final_name)
            except Exception as rollback_exc:
                rollback_errors.append(rollback_exc)
        if rollback_errors:
            raise RuntimeError(
                f"Index promotion failed and rollback was incomplete; preserve Chroma backup {backup_name!r}"
            ) from exc
        raise RuntimeError(f"Index promotion failed for {final_name}; previous indexes were restored") from exc

    if backup_name:
        try:
            client.delete_collection(name=backup_name)
        except Exception:
            # The new pair is committed; an orphan backup is safe and recoverable.
            logger.warning("Could not remove committed Chroma backup collection %s", backup_name, exc_info=True)
    return backup_name


def build_indexes(cfg) -> int:
    """Load non-duplicate chunks into Chroma and BM25; return the indexed count."""
    chunk_file = chunks_path(cfg.kb_sig)
    if not chunk_file.is_file():
        raise FileNotFoundError(f"Chunk file not found for kb_sig={cfg.kb_sig}: {chunk_file}")
    model_name = cfg.kb.get("embed_model")
    if not model_name:
        raise ValueError("kb.embed_model must be set to build indexes")

    chunks = read_jsonl(chunk_file, Chunk)
    indexed_chunks = [chunk for chunk in chunks if not chunk.duplicate_of]
    if not indexed_chunks:
        raise ValueError(f"No non-duplicate chunks available for kb_sig={cfg.kb_sig}")

    # Prepare both new indexes before touching the previous persistent pair.
    texts = [chunk.embed_text for chunk in indexed_chunks]
    vectors = _embed(texts, model_name)
    tokenized_corpus = [tokenize(text) for text in texts]
    from rank_bm25 import BM25Okapi

    try:
        bm25 = BM25Okapi(tokenized_corpus)
    except Exception as exc:
        raise RuntimeError("Could not fit BM25 on the non-duplicate chunk corpus") from exc

    payload = {
        "kb_sig": cfg.kb_sig,
        "chunk_ids": [chunk.chunk_id for chunk in indexed_chunks],
        "tokenized_corpus": tokenized_corpus,
        "bm25": bm25,
    }
    final_bm25 = bm25_path(cfg.kb_sig)
    import chromadb

    staged_bm25 = _write_bm25_staging(final_bm25, payload)
    client = None
    stage_name = _new_collection_name(cfg.collection_name, "stage")
    try:
        client = chromadb.PersistentClient(path=str(PATHS["chroma"]))
        _populate_staging_collection(client, stage_name, indexed_chunks, vectors, model_name)
        _commit_staged_indexes(client, cfg.collection_name, stage_name, staged_bm25, final_bm25)
    except Exception:
        if client is not None:
            try:
                _delete_collection_if_present(client, stage_name)
            except Exception:
                logger.warning("Could not clean staging collection %s", stage_name, exc_info=True)
        staged_bm25.unlink(missing_ok=True)
        raise

    logger.info("Built Chroma and BM25 indexes for kb_sig=%s with %d chunks", cfg.kb_sig, len(indexed_chunks))
    return len(indexed_chunks)
