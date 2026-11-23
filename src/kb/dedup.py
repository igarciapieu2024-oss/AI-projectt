# Mark near-duplicate slide chunks, so the same slide repeated in several decks is indexed only once.
# It reads data/processed/chunks__<kb_sig>.jsonl (written by chunking.py), turns every slide chunk into
# a vector, groups the vectors that are almost identical with DBSCAN (cosine distance), keeps one chunk
# per group and marks the others with duplicate_of. Nothing is deleted: the chunks file is written again
# with the marks, and build_indexes.py leaves the marked chunks out of ChromaDB and BM25.
# Every marked pair is also written to reports/dedup_report__<kb_sig>.csv, to check by hand what was removed.
from __future__ import annotations

import logging

from src.config import PATHS, chunks_path
from src.schemas import Chunk
from src.utils.io import read_jsonl, write_csv, write_jsonl

logger = logging.getLogger(__name__)

# Columns of the report: the group, the chunk that is kept, the chunk marked as its copy and how
# far apart they are (cosine distance; 0 means identical).
REPORT_FIELDS = [
    "cluster_id",
    "representative_id",
    "duplicate_id",
    "cosine_distance",
]


# Mark the near-duplicate slide chunks and return how many were marked.
# Called by the build pipeline (src/pipeline.py) after the chunking and before build_indexes,
# when kb.dedup is true (the default in base.yaml).
def dedup(cfg) -> int:
    settings = cfg.kb
    # eps is the largest cosine distance at which two chunks still count as the same: 0.05 means a
    # cosine similarity of at least 0.95. min_samples 2 means that two chunks are enough to form a group.
    # Both can be changed in the config with kb.dedup_eps and kb.dedup_min_samples.
    eps = float(settings.get("dedup_eps", 0.05))
    min_samples = int(settings.get("dedup_min_samples", 2))
    # The cosine distance goes from 0 to 2, so any other eps is a mistake in the config.
    if not 0 < eps <= 2:
        raise ValueError("kb.dedup_eps must be greater than 0 and at most 2")
    # A group needs at least two chunks: one to keep and one to mark.
    if min_samples < 2:
        raise ValueError("kb.dedup_min_samples must be at least 2")

    # The chunks of this configuration must already exist (chunking.py runs first).
    chunk_file = chunks_path(cfg.kb_sig)
    if not chunk_file.is_file():
        raise FileNotFoundError(f"Chunk file not found for kb_sig={cfg.kb_sig}: {chunk_file}")

    chunks = read_jsonl(chunk_file, Chunk)
    # Only slide chunks are compared: the repeated content is in the slides, while syllabus sections and
    # course summaries are unique and are never marked.
    slide_chunks = [c for c in chunks if c.strategy == "slide" and c.doc_type == "slide"]
    # Clear the marks of a previous run, so running dedup again gives the same result instead of adding
    # marks on top of the old ones.
    for chunk in slide_chunks:
        chunk.duplicate_of = None

    # One row per marked chunk, written to the report at the end.
    report_rows = []
    # With fewer slide chunks than a group needs there is nothing to compare.
    if len(slide_chunks) >= min_samples:
        # The same embedding model as the search (kb.embed_model), so almost identical is measured
        # in the same space the search uses.
        model_name = settings.get("embed_model")
        if not model_name:
            raise ValueError("kb.embed_model must be set to deduplicate chunks")

        # These libraries are heavy, so they are imported only here, when dedup runs: importing this
        # file elsewhere does not load them (a test checks it).
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import DBSCAN
        from sklearn.metrics.pairwise import cosine_distances

        try:
            model = SentenceTransformer(model_name)
            # One vector per slide chunk, from embed_text (course name + position + text), the same text
            # that is indexed. normalize_embeddings gives every vector length 1, as cosine distance expects.
            vectors = model.encode(
                [c.embed_text for c in slide_chunks],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            # The error is logged and raised again, so the build stops instead of indexing duplicates.
            logger.exception("Embedding failed during deduplication with model %s", model_name)
            raise RuntimeError(f"Could not embed slide chunks with model {model_name!r}") from exc

        # One vector per chunk is needed to match each vector back to its chunk.
        if len(vectors) != len(slide_chunks):
            raise ValueError("Embedding model returned a different number of vectors than input chunks")
        # DBSCAN gives every chunk a group number; chunks with no close neighbour get -1.
        # DBSCAN is used because the number of groups is not known in advance.
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(vectors)
        # The cosine distance between every pair of chunks, used below to check each copy directly.
        distances = cosine_distances(vectors)

        # Collect the positions of the chunks of each group. -1 means the chunk has no near-duplicate,
        # so it stays as it is.
        clusters: dict[int, list[int]] = {}
        for index, label in enumerate(labels):
            if label != -1:
                clusters.setdefault(int(label), []).append(index)

        for label, members in sorted(clusters.items()):
            # The chunk with the first id (alphabetical order) is the one kept, so the same chunk is chosen
            # every time the build runs, whatever order DBSCAN returns.
            members.sort(key=lambda i: slide_chunks[i].chunk_id)
            representative_index = members[0]
            representative = slide_chunks[representative_index]
            for index in members[1:]:
                duplicate = slide_chunks[index]
                distance = float(distances[representative_index, index])
                # DBSCAN joins chunks through chains of neighbours (A close to B, B close to C), so a chunk
                # can be in the group without being close to the kept one. It is only marked when it is
                # within eps of the kept chunk itself.
                if distance > eps:
                    continue
                # The mark that build_indexes.py uses to leave this chunk out.
                duplicate.duplicate_of = representative.chunk_id
                report_rows.append({
                    "cluster_id": label,
                    "representative_id": representative.chunk_id,
                    "duplicate_id": duplicate.chunk_id,
                    "cosine_distance": distance,
                })

    # Write all the chunks again (the slide chunks now carry their marks) and the report.
    try:
        write_jsonl(chunk_file, chunks)
        report_path = PATHS["reports"] / f"dedup_report__{cfg.kb_sig}.csv"
        write_csv(report_path, report_rows, REPORT_FIELDS)
    except OSError as exc:
        raise OSError(f"Could not write dedup outputs for kb_sig={cfg.kb_sig}") from exc

    logger.info("Deduplication marked %d chunks from %d slide chunks", len(report_rows), len(slide_chunks))
    # The number of marked chunks, printed by the build pipeline.
    return len(report_rows)
