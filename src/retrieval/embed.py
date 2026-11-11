"""Text -> normalized vectors (model loaded once).

Owner: Sergio.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=3)
def _model(model_name: str):
    """Load a cached local embedding model; never download weights during inference."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Embedding model {model_name!r} is not available locally. "
            "Build/download the model before running offline."
        ) from exc


def embed(texts: list[str], model_name: str):
    """Return a numpy array of normalized embeddings."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("all texts to embed must be strings")
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
    if len(vectors) != len(texts):
        raise ValueError("embedding model returned a different number of vectors than input texts")
    return vectors
