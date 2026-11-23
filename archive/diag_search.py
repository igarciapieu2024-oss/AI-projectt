"""Diagnostic only: shows where the Management Control evaluation chunk is lost. Does not change anything."""
import math
import time

from src.config import load_config
from src.generation.answer import _build_hits
from src.retrieval import bm25, dense, rerank
from src.retrieval.index_cache import load_chunks
from src.config import chunks_path
from src.schemas import Filters, Hit

QUESTION = "What is the final exam worth in Management Control?"
TARGET = "section__mgmt_control_syl_sec08"  # the EVALUATION section of the syllabus

cfg = load_config("configs/final.yaml")
filters = Filters(course_code="mgmt_control")


def show(name, hits):
    # 1. print the list and mark the chunks that hold the answer
    print(f"\n=== {name} ({len(hits)} hits)")
    for i, h in enumerate(hits, 1):
        mark = "  <== ANSWER" if h.chunk.chunk_id.startswith(TARGET) else ""
        print(f"{i:2}. {h.chunk.chunk_id:45} {h.chunk.outline_path[:40]}{mark}")


# 2. each search on its own
show("Meaning search (dense)", dense.search(QUESTION, cfg, filters, cfg.retrieval["k_dense"]))
show("Keyword search (BM25)", bm25.search(QUESTION, cfg, filters, cfg.retrieval["k_bm25"]))
fused = _build_hits(QUESTION, cfg, filters)
show("After fusion (what the reranker receives)", fused)

# 3. reranker score of every fused hit, plus the answer chunks even if they were not fused
model = rerank._model(cfg.retrieval["reranker_model"])
chunks = load_chunks(chunks_path(cfg.kb_sig))
to_score = {h.chunk.chunk_id: h.chunk for h in fused}
for cid, ch in chunks.items():
    if cid.startswith(TARGET):
        to_score[cid] = ch
pairs = [(QUESTION, f"{c.outline_path}\n{c.text}") for c in to_score.values()]
raw = model.predict(pairs, show_progress_bar=False)
print("\n=== Reranker (raw score -> 0-1 score)")
for (cid, c), s in sorted(zip(to_score.items(), raw), key=lambda x: -x[1]):
    mark = "  <== ANSWER" if cid.startswith(TARGET) else ""
    print(f"{float(s):7.2f} -> {1 / (1 + math.exp(-float(s))):.3f}  {cid:45}{mark}")

# 4. the answer chunk text, to see what the reranker reads
print("\n=== Text of the answer chunks")
for cid, c in chunks.items():
    if cid.startswith(TARGET):
        print(f"--- {cid}\n{c.text[:600]}\n")

# 5. router speed: first call (loads the model) vs second call
from src.retrieval.router import route
for i in (1, 2):
    t = time.perf_counter()
    d = route(QUESTION, cfg)
    print(f"Router call {i}: {time.perf_counter() - t:.1f} s  {d}")
