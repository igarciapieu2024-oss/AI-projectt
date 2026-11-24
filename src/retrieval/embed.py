# Turn texts into vectors with the embedding model of the config (all-MiniLM-L6-v2). In the answer phase,
# dense.py uses embed() on the question before searching ChromaDB, and src/warmup.py calls warm_up() when
# the app starts. The chunks were embedded with the same model in build_indexes.py, so questions and chunks
# are compared in the same vector space.
from __future__ import annotations

# lru_cache keeps the loaded model in memory, so it is only loaded the first time.
from functools import lru_cache


# Load the embedding model once and keep it in memory (lru_cache), so we don't load it again for every
# question. local_files_only=True: the model is never downloaded while answering, because the system works
# offline, so it has to be on the laptop already.
@lru_cache(maxsize=3)
def _model(model_name: str):
    try:
        # Imported here, so the library is only loaded when we really need the model.
        from sentence_transformers import SentenceTransformer

        # Load the model from the files on the laptop.
        return SentenceTransformer(model_name, local_files_only=True)
    # If the model can't be loaded from the laptop, we stop with a clear message; "from exc" keeps the
    # original error underneath.
    except Exception as exc:
        raise RuntimeError(
            f"Embedding model {model_name!r} is not available locally. "
            "Build/download the model before running offline."
        ) from exc


# Return one vector per text, normalised to length 1, so the cosine similarity used by ChromaDB only
# compares their direction, which is their meaning.
def embed(texts: list[str], model_name: str):
    # The model name has to be real text.
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    # Every text has to be a string.
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("all texts to embed must be strings")
    # Nothing to embed: we return an empty array without loading the model.
    if not texts:
        import numpy as np

        return np.empty((0, 0), dtype=float)

    try:
        # One vector per text, as a numpy array; normalize_embeddings gives every vector length 1.
        vectors = _model(model_name).encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    # If the model fails, the error says which model it was.
    except Exception as exc:
        raise RuntimeError(f"Could not embed text with local model {model_name!r}") from exc
    # We need one vector per text, to match each vector with its text.
    if len(vectors) != len(texts):
        raise ValueError("embedding model returned a different number of vectors than input texts")
    return vectors


# Embed one short text when the app starts, so the model is loaded before the first question and the first
# answer isn't slower than the rest.
def warm_up(model_name: str) -> None:
    embed(["warm-up"], model_name)
