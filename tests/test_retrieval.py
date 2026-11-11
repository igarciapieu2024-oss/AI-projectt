import math
import pickle
import sys
import types
from types import SimpleNamespace

import pytest

import src.generation.answer as answer_module
import src.retrieval.bm25 as bm25_module
import src.retrieval.dense as dense_module
import src.retrieval.embed as embed_module
import src.retrieval.fusion as fusion_module
import src.retrieval.index_cache as index_cache
import src.retrieval.rerank as rerank_module
from src.schemas import Chunk, Filters, Hit
from src.utils.io import write_jsonl


def _chunk(chunk_id, text, *, course_code="c01", sessions=None):
    return Chunk(
        chunk_id=chunk_id,
        kb_sig="testkb00",
        strategy="slide",
        doc_type="slide",
        course_code=course_code,
        sessions=sessions or [],
        source_id=f"{course_code}_s01",
        covers=[f"{course_code}_s01_p001"],
        parent_id=f"{chunk_id}_parent",
        outline_path=f"Outline for {chunk_id}",
        text=text,
        embed_text=text,
    )


class _Scorer:
    def __init__(self, scores):
        self.scores = scores

    def get_scores(self, tokens):
        assert tokens
        return self.scores


class _Vector:
    def tolist(self):
        return [0.1, 0.2]


def test_embed_uses_cached_local_model_and_normalized_vectors(monkeypatch):
    calls = []

    class FakeModel:
        def __init__(self, name, **kwargs):
            calls.append(("init", name, kwargs))

        def encode(self, texts, **kwargs):
            calls.append(("encode", texts, kwargs))
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeModel),
    )
    embed_module._model.cache_clear()
    try:
        vectors = embed_module.embed(["query"], "local-embedder")
        assert vectors == [[1.0, 0.0]]
        assert calls[0] == ("init", "local-embedder", {"local_files_only": True})
        assert calls[1][2]["normalize_embeddings"] is True
        assert embed_module.embed(["again"], "local-embedder") == [[1.0, 0.0]]
        assert len([call for call in calls if call[0] == "init"]) == 1
    finally:
        embed_module._model.cache_clear()


def test_dense_search_hydrates_chunks_and_applies_exact_session_filter(tmp_path, monkeypatch):
    chunks = [
        _chunk("session_2", "Second session", sessions=[2]),
        _chunk("session_12", "Twelfth session", sessions=[12]),
        _chunk("session_3", "Third session", sessions=[3]),
    ]
    chunk_file = tmp_path / "chunks.jsonl"
    write_jsonl(chunk_file, chunks)
    monkeypatch.setattr(dense_module, "chunks_path", lambda _: chunk_file)
    monkeypatch.setitem(dense_module.PATHS, "chroma", tmp_path / "chroma")
    monkeypatch.setattr(dense_module, "embed", lambda texts, model: [_Vector()])
    calls = []

    class Collection:
        def count(self):
            return 3

        def query(self, **kwargs):
            calls.append(kwargs)
            return {
                "ids": [["session_2", "session_12", "session_3"]],
                "distances": [[0.1, 0.2, 0.3]],
            }

    class Client:
        def list_collections(self):
            return [SimpleNamespace(name="kb_testkb00")]

        def get_collection(self, name):
            assert name == "kb_testkb00"
            return Collection()

    monkeypatch.setitem(
        sys.modules,
        "chromadb",
        SimpleNamespace(PersistentClient=lambda **kwargs: Client()),
    )
    cfg = SimpleNamespace(
        name="test",
        kb_sig="testkb00",
        collection_name="kb_testkb00",
        kb={"embed_model": "local-embedder"},
    )

    hits = dense_module.search("session two", cfg, Filters(sessions=[2]), 2)

    assert [hit.chunk.chunk_id for hit in hits] == ["session_2"]
    assert hits[0].score_dense == pytest.approx(0.9)
    assert calls[0]["n_results"] == 3


def test_dense_search_reports_missing_index(tmp_path, monkeypatch):
    class Client:
        def list_collections(self):
            return []

    monkeypatch.setitem(
        sys.modules,
        "chromadb",
        SimpleNamespace(PersistentClient=lambda **kwargs: Client()),
    )
    cfg = SimpleNamespace(
        name="test",
        kb_sig="testkb00",
        collection_name="kb_testkb00",
        kb={"embed_model": "local-embedder"},
    )
    with pytest.raises(FileNotFoundError, match="Build indexes"):
        dense_module.search("query", cfg, None, 2)


def test_index_cache_reuses_chunks_and_bm25_artifacts(tmp_path, monkeypatch):
    chunk_file = tmp_path / "chunks.jsonl"
    write_jsonl(chunk_file, [_chunk("cached", "Cached chunk")])
    read_calls = []
    original_read = index_cache.read_jsonl

    def counted_read(*args, **kwargs):
        read_calls.append(args[0])
        return original_read(*args, **kwargs)

    monkeypatch.setattr(index_cache, "read_jsonl", counted_read)
    index_cache.clear_caches()
    try:
        first_chunks = index_cache.load_chunks(chunk_file)
        second_chunks = index_cache.load_chunks(chunk_file)
        assert first_chunks is second_chunks
        assert len(read_calls) == 1

        bm25_file = tmp_path / "bm25.pkl"
        with bm25_file.open("wb") as stream:
            pickle.dump({"version": 1}, stream)
        first_bm25 = index_cache.load_bm25(bm25_file)
        second_bm25 = index_cache.load_bm25(bm25_file)
        assert first_bm25 is second_bm25
        assert first_bm25 == {"version": 1}
    finally:
        index_cache.clear_caches()


def test_chroma_client_is_reused_between_collection_lookups(tmp_path, monkeypatch):
    client_instances = []

    class Client:
        def __init__(self):
            self.collection = object()

        def get_collection(self, name):
            return self.collection

    def make_client(**kwargs):
        client = Client()
        client_instances.append(client)
        return client

    monkeypatch.setitem(
        sys.modules,
        "chromadb",
        SimpleNamespace(PersistentClient=make_client),
    )
    index_cache.clear_caches()
    try:
        first = index_cache.get_chroma_collection(tmp_path, "collection")
        second = index_cache.get_chroma_collection(tmp_path, "collection")
        assert first is second
        assert len(client_instances) == 1
    finally:
        index_cache.clear_caches()


def test_bm25_search_loads_index_scores_and_filters_chunks(tmp_path, monkeypatch):
    chunks = [
        _chunk("course_a", "Final exam is 40 percent.", sessions=[2], course_code="c01"),
        _chunk("course_b", "Final exam is 50 percent.", sessions=[2], course_code="c02"),
    ]
    chunk_file = tmp_path / "chunks.jsonl"
    index_file = tmp_path / "bm25.pkl"
    write_jsonl(chunk_file, chunks)
    with index_file.open("wb") as stream:
        pickle.dump({
            "kb_sig": "testkb00",
            "chunk_ids": ["course_a", "course_b"],
            "tokenized_corpus": [["final"], ["final"]],
            "bm25": _Scorer([1.5, 9.0]),
        }, stream)
    monkeypatch.setattr(bm25_module, "chunks_path", lambda _: chunk_file)
    monkeypatch.setattr(bm25_module, "bm25_path", lambda _: index_file)
    cfg = SimpleNamespace(name="test", kb_sig="testkb00")

    hits = bm25_module.search("final exam", cfg, Filters(course_code="c01"), 5)

    assert [(hit.chunk.chunk_id, hit.score_bm25) for hit in hits] == [("course_a", 1.5)]


def test_bm25_rejects_an_index_for_another_kb(tmp_path, monkeypatch):
    index_file = tmp_path / "bm25.pkl"
    with index_file.open("wb") as stream:
        pickle.dump({
            "kb_sig": "different",
            "chunk_ids": [],
            "tokenized_corpus": [],
            "bm25": _Scorer([]),
        }, stream)
    monkeypatch.setattr(bm25_module, "bm25_path", lambda _: index_file)
    with pytest.raises(ValueError, match="kb_sig mismatch"):
        bm25_module.search("query", SimpleNamespace(kb_sig="testkb00"), None, 1)


def test_bm25_does_not_return_zero_score_unmatched_chunks(tmp_path, monkeypatch):
    chunk_file = tmp_path / "chunks.jsonl"
    index_file = tmp_path / "bm25.pkl"
    write_jsonl(chunk_file, [_chunk("unmatched", "Unrelated text.")])
    with index_file.open("wb") as stream:
        pickle.dump({
            "kb_sig": "testkb00",
            "chunk_ids": ["unmatched"],
            "tokenized_corpus": [["unrelated"]],
            "bm25": _Scorer([0.0]),
        }, stream)
    monkeypatch.setattr(bm25_module, "chunks_path", lambda _: chunk_file)
    monkeypatch.setattr(bm25_module, "bm25_path", lambda _: index_file)

    assert bm25_module.search("unrelated", SimpleNamespace(kb_sig="testkb00"), None, 3) == []


def test_rrf_combines_ranked_lists_without_mutating_inputs():
    first = Hit(chunk=_chunk("a", "A"), rank=1, score_dense=0.9)
    second = Hit(chunk=_chunk("b", "B"), rank=2, score_dense=0.8)
    keyword_first = Hit(chunk=_chunk("b", "B"), rank=1, score_bm25=5.0)
    keyword_second = Hit(chunk=_chunk("a", "A"), rank=2, score_bm25=4.0)

    fused = fusion_module.rrf([[first, second], [keyword_first, keyword_second]], k=10)

    assert [hit.chunk.chunk_id for hit in fused] == ["a", "b"]
    assert fused[0].score_rrf == pytest.approx(1 / 11 + 1 / 12)
    assert fused[0].score_dense == 0.9
    assert fused[0].score_bm25 == 4.0
    assert first.score_rrf is None


def test_rerank_sorts_cross_encoder_scores_and_preserves_retrieval_scores(monkeypatch):
    calls = []

    class FakeCrossEncoder:
        def __init__(self, name, **kwargs):
            calls.append(("init", name, kwargs))

        def predict(self, pairs, **kwargs):
            calls.append(("predict", pairs, kwargs))
            return [0.1, 0.9]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )
    rerank_module._model.cache_clear()
    try:
        hits = [
            Hit(chunk=_chunk("a", "First evidence"), rank=1, score_rrf=0.02),
            Hit(chunk=_chunk("b", "Better evidence"), rank=2, score_rrf=0.01),
        ]
        cfg = SimpleNamespace(retrieval={"reranker_model": "local-reranker"})

        ranked = rerank_module.rerank("question", hits, 1, cfg)

        assert [hit.chunk.chunk_id for hit in ranked] == ["b"]
        assert ranked[0].score_rerank == pytest.approx(1 / (1 + math.exp(-0.9)))
        assert ranked[0].score_rrf == 0.01
        assert calls[0][2]["automodel_args"] == {"local_files_only": True}
    finally:
        rerank_module._model.cache_clear()


def test_answer_hybrid_retrieval_fuses_and_reranks(monkeypatch):
    dense_hit = Hit(chunk=_chunk("dense", "Dense hit"), rank=1, score_dense=0.8)
    bm25_hit = Hit(chunk=_chunk("keyword", "Keyword hit"), rank=1, score_bm25=2.0)
    fused = [
        Hit(chunk=dense_hit.chunk, rank=1, score_rrf=0.03),
        Hit(chunk=bm25_hit.chunk, rank=2, score_rrf=0.02),
    ]
    calls = []
    monkeypatch.setattr(answer_module.dense, "search", lambda *args: calls.append("dense") or [dense_hit])
    monkeypatch.setattr(answer_module.bm25, "search", lambda *args: calls.append("bm25") or [bm25_hit])
    monkeypatch.setattr(answer_module.fusion, "rrf", lambda lists, k: calls.append(("rrf", k)) or fused)
    monkeypatch.setattr(
        answer_module.rerank,
        "rerank",
        lambda query, hits, top_k, cfg: calls.append(("rerank", top_k)) or hits[:top_k],
    )
    monkeypatch.setattr(answer_module, "chat", lambda messages, cfg: "Answer [S1]")
    cfg = SimpleNamespace(
        name="test",
        kb_sig="testkb00",
        retrieval={
            "mode": "hybrid", "top_k": 1, "k_dense": 4, "k_bm25": 6, "rrf_k": 30, "rerank": True,
        },
        generation={"rag": True, "prompt_version": "v2"},
    )

    result = answer_module.answer("query", cfg=cfg)

    assert calls == ["dense", "bm25", ("rrf", 30), ("rerank", 1)]
    assert result.context_hits[0].label == "S1"
    assert result.timings_ms["rerank"] >= 0


def test_dense_only_mode_skips_keyword_search(monkeypatch):
    dense_hit = Hit(chunk=_chunk("dense", "Dense hit"), rank=1, score_dense=0.8)
    monkeypatch.setattr(answer_module.dense, "search", lambda *args: [dense_hit])
    monkeypatch.setattr(
        answer_module.bm25,
        "search",
        lambda *args: pytest.fail("dense mode must not run BM25"),
    )
    cfg = SimpleNamespace(
        retrieval={"mode": "dense", "top_k": 1, "k_dense": 4},
    )

    assert answer_module._build_hits("query", cfg, None) == [dense_hit]
