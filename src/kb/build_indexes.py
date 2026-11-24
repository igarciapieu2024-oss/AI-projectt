# Build the two indexes the search uses, from the chunks that aren't marked as duplicates:
#   ChromaDB (folder chroma_db/): the vector of every chunk, for the dense search (search by meaning), in a collection
# named after the kb_sig of the configuration.
#   BM25 (data/processed/bm25__<kb_sig>.pkl): a keyword index, for the search by exact words.
# Both are built from embed_text (course name + position + text). src/retrieval/dense.py and src/retrieval/bm25.py
# read them in the answer phase.
# We first build and check the new indexes under temporary names, and only then let them replace the old ones, so a
# build that fails half way never leaves the app with a broken or half-written index.
from __future__ import annotations

# logging for messages; os and tempfile to write files safely; pickle to save the BM25 index; re for the tokenizer;
# uuid for random temporary names.
import logging
import os
import pickle
import re
import tempfile
import uuid
from pathlib import Path

# Where the indexes and the chunks are saved, the format of a chunk and the JSONL reader.
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

# Split a text into lower-case BM25 tokens. src/retrieval/bm25.py uses the same function on the questions, so
# questions and chunks are split in exactly the same way.
def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall((text or "").lower())


# The fields of a chunk that we store in ChromaDB next to its vector. ChromaDB metadata only accepts single values
# (text, numbers, true/false), so we turn lists into text. course_code is the field the course filter uses in the
# dense search.
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


# Rebuild a complete Chunk from its id by reading the chunks file of this knowledge base. ChromaDB only stores the
# metadata above and embed_text, so fields such as text and covers have to come from the chunks file. The answer phase
# doesn't use this function: dense.py and bm25.py read the chunks file once and keep it in memory, which is faster
# than reading it for every result.
def load_chunk_by_id(chunk_id: str, cfg) -> Chunk:
    # The chunks file of this knowledge base has to exist.
    chunk_file = chunks_path(cfg.kb_sig)
    if not chunk_file.is_file():
        raise FileNotFoundError(f"Chunk file not found for kb_sig={cfg.kb_sig}: {chunk_file}")
    # Read the whole file and keep the chunks with this id.
    matches = [chunk for chunk in read_jsonl(chunk_file, Chunk) if chunk.chunk_id == chunk_id]
    # Exactly one chunk should have this id; none or several means the file is wrong.
    if not matches:
        raise KeyError(f"Chunk ID {chunk_id!r} not found for kb_sig={cfg.kb_sig}")
    if len(matches) > 1:
        raise ValueError(f"Chunk ID {chunk_id!r} occurs more than once for kb_sig={cfg.kb_sig}")
    return matches[0]


# Turn the texts into vectors with the embedding model of the config (all-MiniLM-L6-v2, 384 numbers). We normalise the
# vectors to length 1, so the cosine similarity used by ChromaDB only compares their direction, which is what carries
# the meaning.
def _embed(texts: list[str], model_name: str):
    # Imported here, because the library is heavy and only this step needs it.
    from sentence_transformers import SentenceTransformer

    try:
        # Load the model and turn every text into a vector (a numpy array, without a progress bar).
        model = SentenceTransformer(model_name)
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    # If the model fails, we log the error and stop the build with a clear message.
    except Exception as exc:
        logger.exception("Embedding failed while building indexes with model %s", model_name)
        raise RuntimeError(f"Could not embed chunks with model {model_name!r}") from exc
    # We need one vector per text, to store each vector with its chunk.
    if len(vectors) != len(texts):
        raise ValueError("Embedding model returned a different number of vectors than input chunks")
    return vectors


# ----------------------------------------------------------------------
# Build the new indexes under temporary names

# A temporary name for a ChromaDB collection: the final name plus the kind ("stage" or "backup") and a random code, so
# it never clashes with an existing collection. ChromaDB names can have at most 63 characters, so we shorten the base
# name if needed.
def _new_collection_name(base: str, kind: str) -> str:
    # The ending: the kind and 10 random characters.
    suffix = f"__{kind}_{uuid.uuid4().hex[:10]}"
    return f"{base[: 63 - len(suffix)]}{suffix}"


# The names of all the collections in the ChromaDB folder. Different ChromaDB versions return either collection
# objects or plain names, so we handle both.
def _collection_names(client) -> set[str]:
    return {item.name if hasattr(item, "name") else str(item) for item in client.list_collections()}


# Delete a collection only if it exists; we use it to clean up a temporary collection after an error.
def _delete_collection_if_present(client, name: str) -> None:
    if name in _collection_names(client):
        client.delete_collection(name=name)


# Create the temporary ChromaDB collection, add every chunk and check that all of them were stored.
def _populate_staging_collection(client, name: str, chunks: list[Chunk], vectors, model_name: str) -> None:
    collection = client.create_collection(
        name=name,
        # We save the kb_sig and the embedding model with the collection, so it's clear how it was built.
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
    # The check: the collection has to hold exactly the chunks we added, no more and no fewer.
    # Read back what the collection stored and compare its ids with the ones we added.
    stored = collection.get(limit=len(chunks), include=["metadatas"])
    expected_ids = [chunk.chunk_id for chunk in chunks]
    if collection.count() != len(chunks) or set(stored["ids"]) != set(expected_ids):
        raise RuntimeError(f"Staging collection {name!r} failed its count/ID validation")


# Write the BM25 index to a temporary file in the same folder as the final one, read it back and check it. It returns
# the path of the temporary file.
def _write_bm25_staging(path: Path, payload: dict) -> Path:
    # Make sure the folder exists.
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        # pickle saves the whole Python object (the BM25 model and its corpus) to a file.
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.stage.", delete=False) as stream:
            temp_path = Path(stream.name)
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            # flush and fsync make sure the data is really on the disk before we read it back.
            stream.flush()
            os.fsync(stream.fileno())
        # We read the file back and check its keys, and that the chunk ids and the tokens are in the same order,
        # because BM25 finds each chunk by its position in these lists.
        with temp_path.open("rb") as stream:
            check = pickle.load(stream)
        if set(check) != _BM25_KEYS:
            raise ValueError("Staged BM25 pickle has unexpected keys")
        if check["chunk_ids"] != payload["chunk_ids"] or check["tokenized_corpus"] != payload["tokenized_corpus"]:
            raise ValueError("Staged BM25 pickle failed corpus-order validation")
        return temp_path
    except Exception:
        # Any error: we delete the temporary file, so no half-written index is left behind.
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        logger.exception("Could not stage BM25 index for %s", path)
        raise


# ----------------------------------------------------------------------
# Replace the old indexes with the new ones

# Swap in the checked temporary indexes: the old ChromaDB collection is renamed as a backup, the new one takes the
# final name, and the new BM25 file replaces the old one. If any step fails, we undo the changes already made, so
# ChromaDB and BM25 always belong to the same build. It returns the backup name, if there is one. If the computer
# crashes in the middle of the swap, the backup or the temporary collection can be left behind, so the index can be
# recovered by hand.
def _commit_staged_indexes(client, final_name: str, stage_name: str, staged_bm25: Path, final_bm25: Path) -> str | None:
    # The collections that exist right now.
    names = _collection_names(client)
    # We only need a backup if there's already a collection with the final name.
    had_previous = final_name in names
    backup_name = _new_collection_name(final_name, "backup") if had_previous else None
    # What we've done so far, so we know what to undo if something fails.
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
        # 3. Replace the BM25 file in one operation, so it's never half written.
        os.replace(staged_bm25, final_bm25)
    except Exception as exc:
        logger.exception("Index promotion failed for %s; attempting rollback", final_name)
        rollback_errors = []
        # Undo step 2: the new collection goes back to its temporary name.
        if stage_promoted:
            try:
                client.get_collection(name=final_name).modify(name=stage_name)
            except Exception as rollback_exc:
                # We keep the backup, so the old index can still be recovered by hand.
                rollback_errors.append(rollback_exc)
        # Undo step 1: the old collection gets its final name back.
        if backup_renamed and backup_name:
            try:
                client.get_collection(name=backup_name).modify(name=final_name)
            except Exception as rollback_exc:
                rollback_errors.append(rollback_exc)
        # The error says whether the old indexes were restored or the backup has to be recovered by hand.
        if rollback_errors:
            raise RuntimeError(
                f"Index promotion failed and rollback was incomplete; preserve Chroma backup {backup_name!r}"
            ) from exc
        raise RuntimeError(f"Index promotion failed for {final_name}; previous indexes were restored") from exc

    # The swap worked, so we no longer need the backup of the old collection.
    if backup_name:
        try:
            client.delete_collection(name=backup_name)
        except Exception:
            # The new indexes are already in place; an old backup left behind does no harm.
            logger.warning("Could not remove committed Chroma backup collection %s", backup_name, exc_info=True)
    return backup_name


# ----------------------------------------------------------------------
# Main function

# Build ChromaDB and BM25 from the chunks that aren't marked as duplicates and return how many were indexed. The build
# pipeline (src/pipeline.py) calls it as its last step, after dedup.
def build_indexes(cfg) -> int:
    # The chunks of this configuration (chunking.py) with their duplicate marks (dedup.py).
    chunk_file = chunks_path(cfg.kb_sig)
    if not chunk_file.is_file():
        raise FileNotFoundError(f"Chunk file not found for kb_sig={cfg.kb_sig}: {chunk_file}")
    # The embedding model has to be set in the config.
    model_name = cfg.kb.get("embed_model")
    if not model_name:
        raise ValueError("kb.embed_model must be set to build indexes")

    # Read every chunk.
    chunks = read_jsonl(chunk_file, Chunk)
    # We leave out the chunks marked as near-duplicates; the rest are the ones the search can return.
    indexed_chunks = [chunk for chunk in chunks if not chunk.duplicate_of]
    # With nothing to index we stop.
    if not indexed_chunks:
        raise ValueError(f"No non-duplicate chunks available for kb_sig={cfg.kb_sig}")

    # We prepare both new indexes completely before touching the existing ones. Both are built from embed_text, so
    # both searches see the course name and the position of each chunk.
    texts = [chunk.embed_text for chunk in indexed_chunks]
    # Dense index: one vector per chunk.
    vectors = _embed(texts, model_name)
    # The keyword index: every chunk split into tokens, and then the BM25 model fitted on all of them. BM25Okapi
    # scores a chunk by the question words it contains, and gives more weight to rare words.
    tokenized_corpus = [tokenize(text) for text in texts]
    # Imported here, so the library is only loaded when the indexes are built.
    from rank_bm25 import BM25Okapi

    try:
        bm25 = BM25Okapi(tokenized_corpus)
    except Exception as exc:
        raise RuntimeError("Could not fit BM25 on the non-duplicate chunk corpus") from exc

    # Everything bm25.py needs: the kb_sig (checked when the file is loaded), the chunk ids in the same order as the
    # tokens, the tokens and the fitted model.
    payload = {
        "kb_sig": cfg.kb_sig,
        "chunk_ids": [chunk.chunk_id for chunk in indexed_chunks],
        "tokenized_corpus": tokenized_corpus,
        "bm25": bm25,
    }
    # Where the final BM25 file goes.
    final_bm25 = bm25_path(cfg.kb_sig)
    import chromadb

    # Write and check the temporary BM25 file, then build the temporary ChromaDB collection and swap both in.
    staged_bm25 = _write_bm25_staging(final_bm25, payload)
    client = None
    # The temporary name of the new ChromaDB collection.
    stage_name = _new_collection_name(cfg.collection_name, "stage")
    try:
        # PersistentClient saves the collections on disk, in the chroma_db/ folder.
        client = chromadb.PersistentClient(path=str(PATHS["chroma"]))
        _populate_staging_collection(client, stage_name, indexed_chunks, vectors, model_name)
        _commit_staged_indexes(client, cfg.collection_name, stage_name, staged_bm25, final_bm25)
    except Exception:
        # Any error: we remove the temporary collection and the temporary BM25 file, and the old indexes stay as they
        # were.
        if client is not None:
            try:
                _delete_collection_if_present(client, stage_name)
            except Exception:
                logger.warning("Could not clean staging collection %s", stage_name, exc_info=True)
        staged_bm25.unlink(missing_ok=True)
        raise

    # Write in the log how many chunks were indexed.
    logger.info("Built Chroma and BM25 indexes for kb_sig=%s with %d chunks", cfg.kb_sig, len(indexed_chunks))
    # The number of indexed chunks, printed by the build pipeline.
    return len(indexed_chunks)
