# Build the two indexes the search uses, from the chunks that are not marked as duplicates:
#   ChromaDB (folder chroma_db/): the vector of every chunk, for dense search (search by meaning),
#     in a collection named after the kb_sig of the configuration.
#   BM25 (data/processed/bm25__<kb_sig>.pkl): a keyword index, for search by exact words.
# Both are built from embed_text (course name + position + text). src/retrieval/dense.py and
# src/retrieval/bm25.py read them in the answer phase.
# The new indexes are first built and checked under temporary names, and only then replace the old
# ones, so a build that fails half way never leaves the app with a broken or half-written index.
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

# A token for BM25: a run of letters or digits (the underscore is excluded), or the % sign on its own,
# so percentages in a question can match the percentages of the syllabi.
_TOKEN_PATTERN = re.compile(r"[^\W_]+|%", flags=re.UNICODE)
# The four parts the BM25 file must contain; bm25.py checks the same keys when it loads the file.
_BM25_KEYS = {"kb_sig", "chunk_ids", "tokenized_corpus", "bm25"}


# ----------------------------------------------------------------------
# Helpers

# Split a text into lower-case BM25 tokens. The same function is used on the questions by
# src/retrieval/bm25.py, so questions and chunks are split in exactly the same way.
def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall((text or "").lower())


# The fields of a chunk stored in ChromaDB next to its vector. ChromaDB metadata only accepts single
# values (text, numbers, true/false), so lists are turned into text. course_code is the field the
# course filter uses in the dense search.
def _metadata(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "kb_sig": chunk.kb_sig,
        "strategy": chunk.strategy,
        "doc_type": chunk.doc_type,
        "course_code": chunk.course_code,
        "source_id": chunk.source_id,
        # The session list becomes text, with the numbers separated by semicolons.
        "sessions": ";".join(str(value) for value in chunk.sessions),
        "parent_id": chunk.parent_id,
        "outline_path": chunk.outline_path,
        "has_image_text": chunk.has_image_text,
    }


# Rebuild a complete Chunk from its id by reading the chunks file of this knowledge base.
# ChromaDB stores only the metadata above and embed_text, so fields such as text and covers have to be
# read from the chunks file. Not used by the answer phase: dense.py and bm25.py read the chunks file
# once and keep it in memory, which is faster than reading it for every result.
def load_chunk_by_id(chunk_id: str, cfg) -> Chunk:
    chunk_file = chunks_path(cfg.kb_sig)
    if not chunk_file.is_file():
        raise FileNotFoundError(f"Chunk file not found for kb_sig={cfg.kb_sig}: {chunk_file}")
    matches = [chunk for chunk in read_jsonl(chunk_file, Chunk) if chunk.chunk_id == chunk_id]
    # Exactly one chunk must have this id; none or several means the file is wrong.
    if not matches:
        raise KeyError(f"Chunk ID {chunk_id!r} not found for kb_sig={cfg.kb_sig}")
    if len(matches) > 1:
        raise ValueError(f"Chunk ID {chunk_id!r} occurs more than once for kb_sig={cfg.kb_sig}")
    return matches[0]


# Turn the texts into vectors with the embedding model of the config (all-MiniLM-L6-v2, 384 numbers).
# The vectors are normalised to length 1, so the cosine similarity used by ChromaDB compares only
# their direction, which is what carries the meaning.
def _embed(texts: list[str], model_name: str):
    # Imported here because the library is heavy and only this step needs it.
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
    # One vector per text is needed to store each vector with its chunk.
    if len(vectors) != len(texts):
        raise ValueError("Embedding model returned a different number of vectors than input chunks")
    return vectors


# ----------------------------------------------------------------------
# Build the new indexes under temporary names

# A temporary ChromaDB collection name: the final name plus the kind ("stage" or "backup") and a random
# code, so it never clashes with an existing collection. ChromaDB names have at most 63 characters, so
# the base name is shortened if needed.
def _new_collection_name(base: str, kind: str) -> str:
    suffix = f"__{kind}_{uuid.uuid4().hex[:10]}"
    return f"{base[: 63 - len(suffix)]}{suffix}"


# The names of all the collections in the ChromaDB folder. Different ChromaDB versions return either
# collection objects or plain names, so both are handled.
def _collection_names(client) -> set[str]:
    return {item.name if hasattr(item, "name") else str(item) for item in client.list_collections()}


# Delete a collection only if it exists, used to clean up a temporary collection after an error.
def _delete_collection_if_present(client, name: str) -> None:
    if name in _collection_names(client):
        client.delete_collection(name=name)


# Create the temporary ChromaDB collection, add every chunk and check that all of them were stored.
def _populate_staging_collection(client, name: str, chunks: list[Chunk], vectors, model_name: str) -> None:
    collection = client.create_collection(
        name=name,
        # The kb_sig and the embedding model are saved with the collection, so it is clear how it was built.
        # "hnsw:space": "cosine" makes ChromaDB measure distances with the cosine, like the rest of the project.
        metadata={
            "kb_sig": chunks[0].kb_sig,
            "embed_model": model_name,
            "hnsw:space": "cosine",
        },
    )
    # One entry per chunk: its id, its vector, its embed_text and its metadata.
    collection.add(
        ids=[chunk.chunk_id for chunk in chunks],
        embeddings=vectors.tolist(),
        documents=[chunk.embed_text for chunk in chunks],
        metadatas=[_metadata(chunk) for chunk in chunks],
    )
    # Check: the collection must hold exactly the chunks that were added, no more and no fewer.
    stored = collection.get(limit=len(chunks), include=["metadatas"])
    expected_ids = [chunk.chunk_id for chunk in chunks]
    if collection.count() != len(chunks) or set(stored["ids"]) != set(expected_ids):
        raise RuntimeError(f"Staging collection {name!r} failed its count/ID validation")


# Write the BM25 index to a temporary file in the same folder as the final one, read it back and check it.
# Returns the path of the temporary file.
def _write_bm25_staging(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        # pickle saves the whole Python object (the BM25 model and its corpus) to a file.
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.stage.", delete=False) as stream:
            temp_path = Path(stream.name)
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            # flush and fsync make sure the data is really on the disk before it is read back.
            stream.flush()
            os.fsync(stream.fileno())
        # Read the file back and check its keys and that the chunk ids and tokens are in the same order,
        # because BM25 finds each chunk by its position in these lists.
        with temp_path.open("rb") as stream:
            check = pickle.load(stream)
        if set(check) != _BM25_KEYS:
            raise ValueError("Staged BM25 pickle has unexpected keys")
        if check["chunk_ids"] != payload["chunk_ids"] or check["tokenized_corpus"] != payload["tokenized_corpus"]:
            raise ValueError("Staged BM25 pickle failed corpus-order validation")
        return temp_path
    except Exception:
        # Any error: delete the temporary file so no half-written index is left behind.
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        logger.exception("Could not stage BM25 index for %s", path)
        raise


# ----------------------------------------------------------------------
# Replace the old indexes with the new ones

# Swap the checked temporary indexes in: the old ChromaDB collection is renamed as a backup, the new one
# takes the final name, and the new BM25 file replaces the old one. If any step fails, the changes already
# made are undone, so ChromaDB and BM25 always belong to the same build. Returns the backup name, if any.
# A crash of the computer in the middle of the swap can leave the backup or temporary collection behind,
# so it can be recovered by hand.
def _commit_staged_indexes(client, final_name: str, stage_name: str, staged_bm25: Path, final_bm25: Path) -> str | None:
    names = _collection_names(client)
    # A backup is only needed if there is already a collection with the final name.
    had_previous = final_name in names
    backup_name = _new_collection_name(final_name, "backup") if had_previous else None
    # What has been done so far, to know what to undo if something fails.
    backup_renamed = False
    stage_promoted = False
    try:
        # 1. Move the old collection out of the way under the backup name.
        if backup_name:
            client.get_collection(name=final_name).modify(name=backup_name)
            backup_renamed = True
        # 2. Give the new collection the final name.
        client.get_collection(name=stage_name).modify(name=final_name)
        stage_promoted = True
        # 3. Replace the BM25 file in one operation, so it is never half written.
        os.replace(staged_bm25, final_bm25)
    except Exception as exc:
        logger.exception("Index promotion failed for %s; attempting rollback", final_name)
        rollback_errors = []
        # Undo step 2: the new collection goes back to its temporary name.
        if stage_promoted:
            try:
                client.get_collection(name=final_name).modify(name=stage_name)
            except Exception as rollback_exc:
                # The backup is kept, so the old index can still be recovered by hand.
                rollback_errors.append(rollback_exc)
        # Undo step 1: the old collection gets its final name back.
        if backup_renamed and backup_name:
            try:
                client.get_collection(name=backup_name).modify(name=final_name)
            except Exception as rollback_exc:
                rollback_errors.append(rollback_exc)
        # The error says whether the old indexes were restored or the backup must be recovered by hand.
        if rollback_errors:
            raise RuntimeError(
                f"Index promotion failed and rollback was incomplete; preserve Chroma backup {backup_name!r}"
            ) from exc
        raise RuntimeError(f"Index promotion failed for {final_name}; previous indexes were restored") from exc

    # The swap worked: the backup of the old collection is no longer needed.
    if backup_name:
        try:
            client.delete_collection(name=backup_name)
        except Exception:
            # The new indexes are already in place; an old backup left behind does no harm.
            logger.warning("Could not remove committed Chroma backup collection %s", backup_name, exc_info=True)
    return backup_name


# ----------------------------------------------------------------------
# Main function

# Build ChromaDB and BM25 from the chunks not marked as duplicates and return how many were indexed.
# Called by the build pipeline (src/pipeline.py) as its last step, after dedup.
def build_indexes(cfg) -> int:
    # The chunks of this configuration (chunking.py) with their duplicate marks (dedup.py).
    chunk_file = chunks_path(cfg.kb_sig)
    if not chunk_file.is_file():
        raise FileNotFoundError(f"Chunk file not found for kb_sig={cfg.kb_sig}: {chunk_file}")
    model_name = cfg.kb.get("embed_model")
    if not model_name:
        raise ValueError("kb.embed_model must be set to build indexes")

    chunks = read_jsonl(chunk_file, Chunk)
    # Leave out the chunks marked as near-duplicates; the rest are the ones the search can return.
    indexed_chunks = [chunk for chunk in chunks if not chunk.duplicate_of]
    if not indexed_chunks:
        raise ValueError(f"No non-duplicate chunks available for kb_sig={cfg.kb_sig}")

    # Prepare both new indexes completely before touching the existing ones.
    # Both are built from embed_text, so both searches see the course name and the position of each chunk.
    texts = [chunk.embed_text for chunk in indexed_chunks]
    # Dense index: one vector per chunk.
    vectors = _embed(texts, model_name)
    # Keyword index: every chunk split into tokens, then the BM25 model fitted on all of them.
    # BM25Okapi scores a chunk by the question words it contains, giving more weight to rare words.
    tokenized_corpus = [tokenize(text) for text in texts]
    from rank_bm25 import BM25Okapi

    try:
        bm25 = BM25Okapi(tokenized_corpus)
    except Exception as exc:
        raise RuntimeError("Could not fit BM25 on the non-duplicate chunk corpus") from exc

    # Everything bm25.py needs: the kb_sig (checked when the file is loaded), the chunk ids in the same order
    # as the tokens, the tokens and the fitted model.
    payload = {
        "kb_sig": cfg.kb_sig,
        "chunk_ids": [chunk.chunk_id for chunk in indexed_chunks],
        "tokenized_corpus": tokenized_corpus,
        "bm25": bm25,
    }
    final_bm25 = bm25_path(cfg.kb_sig)
    import chromadb

    # Write and check the temporary BM25 file, then build the temporary ChromaDB collection and swap both in.
    staged_bm25 = _write_bm25_staging(final_bm25, payload)
    client = None
    stage_name = _new_collection_name(cfg.collection_name, "stage")
    try:
        # PersistentClient saves the collections on disk, in the chroma_db/ folder.
        client = chromadb.PersistentClient(path=str(PATHS["chroma"]))
        _populate_staging_collection(client, stage_name, indexed_chunks, vectors, model_name)
        _commit_staged_indexes(client, cfg.collection_name, stage_name, staged_bm25, final_bm25)
    except Exception:
        # Any error: remove the temporary collection and the temporary BM25 file; the old indexes stay as they were.
        if client is not None:
            try:
                _delete_collection_if_present(client, stage_name)
            except Exception:
                logger.warning("Could not clean staging collection %s", stage_name, exc_info=True)
        staged_bm25.unlink(missing_ok=True)
        raise

    logger.info("Built Chroma and BM25 indexes for kb_sig=%s with %d chunks", cfg.kb_sig, len(indexed_chunks))
    # The number of indexed chunks, printed by the build pipeline.
    return len(indexed_chunks)
