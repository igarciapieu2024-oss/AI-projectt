"""Mark near-duplicate slide chunks with DBSCAN using cosine distance."""
from __future__ import annotations

import logging

from src.config import PATHS, chunks_path
from src.schemas import Chunk
from src.utils.io import read_jsonl, write_csv, write_jsonl

logger = logging.getLogger(__name__)

REPORT_FIELDS = [
    "cluster_id",
    "representative_id",
    "duplicate_id",
    "cosine_distance",
]


def dedup(cfg) -> int:
    """Mark duplicate slide chunks and return the number marked.

    DBSCAN defaults to a maximum neighbor distance of 0.05 (cosine similarity
    0.95) and requires at least two members to form a cluster. Both values can
    be overridden with ``kb.dedup_eps`` and ``kb.dedup_min_samples``.
    """
    settings = cfg.kb
    eps = float(settings.get("dedup_eps", 0.05))
    min_samples = int(settings.get("dedup_min_samples", 2))
    if not 0 < eps <= 2:
        raise ValueError("kb.dedup_eps must be greater than 0 and at most 2")
    if min_samples < 2:
        raise ValueError("kb.dedup_min_samples must be at least 2")

    chunk_file = chunks_path(cfg.kb_sig)
    if not chunk_file.is_file():
        raise FileNotFoundError(f"Chunk file not found for kb_sig={cfg.kb_sig}: {chunk_file}")

    chunks = read_jsonl(chunk_file, Chunk)
    slide_chunks = [c for c in chunks if c.strategy == "slide" and c.doc_type == "slide"]
    # Clear old annotations on eligible chunks so rerunning dedup is idempotent.
    for chunk in slide_chunks:
        chunk.duplicate_of = None

    report_rows = []
    if len(slide_chunks) >= min_samples:
        model_name = settings.get("embed_model")
        if not model_name:
            raise ValueError("kb.embed_model must be set to deduplicate chunks")

        # Keep heavyweight dependencies lazy: importing this module must not load
        # sklearn or sentence-transformers (the import-contract tests rely on it).
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import DBSCAN
        from sklearn.metrics.pairwise import cosine_distances

        try:
            model = SentenceTransformer(model_name)
            vectors = model.encode(
                [c.embed_text for c in slide_chunks],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.exception("Embedding failed during deduplication with model %s", model_name)
            raise RuntimeError(f"Could not embed slide chunks with model {model_name!r}") from exc

        if len(vectors) != len(slide_chunks):
            raise ValueError("Embedding model returned a different number of vectors than input chunks")
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(vectors)
        distances = cosine_distances(vectors)

        clusters: dict[int, list[int]] = {}
        for index, label in enumerate(labels):
            if label != -1:  # DBSCAN noise is deliberately left untouched.
                clusters.setdefault(int(label), []).append(index)

        for label, members in sorted(clusters.items()):
            # Sort by ID so representative choice is independent of cluster order.
            members.sort(key=lambda i: slide_chunks[i].chunk_id)
            representative_index = members[0]
            representative = slide_chunks[representative_index]
            for index in members[1:]:
                duplicate = slide_chunks[index]
                distance = float(distances[representative_index, index])
                # DBSCAN clusters connected neighborhoods, so cluster membership
                # alone does not guarantee direct similarity to the representative.
                if distance > eps:
                    continue
                duplicate.duplicate_of = representative.chunk_id
                report_rows.append({
                    "cluster_id": label,
                    "representative_id": representative.chunk_id,
                    "duplicate_id": duplicate.chunk_id,
                    "cosine_distance": distance,
                })

    try:
        write_jsonl(chunk_file, chunks)
        report_path = PATHS["reports"] / f"dedup_report__{cfg.kb_sig}.csv"
        write_csv(report_path, report_rows, REPORT_FIELDS)
    except OSError as exc:
        raise OSError(f"Could not write dedup outputs for kb_sig={cfg.kb_sig}") from exc

    logger.info("Deduplication marked %d chunks from %d slide chunks", len(report_rows), len(slide_chunks))
    return len(report_rows)
