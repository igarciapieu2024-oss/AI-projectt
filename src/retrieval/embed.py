# Turn texts into vectors with the embedding model of the config (all-MiniLM-L6-v2). Used in the answer
# phase: dense.py embeds the question with embed() before searching ChromaDB, and src/warmup.py calls
# warm_up() when the app starts. The chunks were embedded with the same model in build_indexes.py, so
# questions and chunks are compared in the same vector space.
from __future__ import annotations

from functools import lru_cache


# Load the embedding model once and keep it in memory (lru_cache), so it is not loaded again for every
# question. local_files_only=True: the model is never downloaded while answering, because the system
# works offline; it must already be on the laptop.
@lru_cache(maxsize=3)
def _model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Embedding model {model_name!r} is not available locally. "
            "Build/download the model before running offline."
        ) from exc


# Return one vector per text, normalised to length 1, so the cosine similarity used by ChromaDB
# compares only their direction (their meaning).
def embed(texts: list[str], model_name: str):
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("all texts to embed must be strings")
    # Nothing to embed: an empty array, without loading the model.
    if not texts:
        import numpy as np

        return np.empty((0, 0), dtype=float)

    try:
        vectors = _model(model_name).encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not embed text with local model {model_name!r}") from exc
    # One vector per text is needed to match each vector with its text.
    if len(vectors) != len(texts):
        raise ValueError("embedding model returned a different number of vectors than input texts")
    return vectors


# Embed one short text when the app starts, so the model is loaded before the first question and the
# first answer is not slower than the rest.
def warm_up(model_name: str) -> None:
    embed(["warm-up"], model_name)
