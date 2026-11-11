import pickle
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from src.kb import build_indexes
from src.schemas import Chunk
from src.utils.io import write_jsonl


def _chunk(chunk_id, text, duplicate_of=None, sessions=None):
    return Chunk(
        chunk_id=chunk_id, kb_sig="testkb00", strategy="slide", doc_type="slide",
        course_code="c01", sessions=sessions or [2, 3], source_id="c01_s02-03",
        covers=["c01_s02-03_p001"], parent_id="parent1", outline_path="Session 2",
        text=text, embed_text=f"Session 2\n{text}", has_image_text=False,
        duplicate_of=duplicate_of,
    )


def _fake_embeddings(monkeypatch):
    class FakeModel:
        def __init__(self, name):
            assert name == "test-model"

        def encode(self, texts, **kwargs):
            return np.array([
                [0.0, 1.0] if "Attendance" in text else [1.0, 0.0]
                for text in texts
            ])

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeModel))


def test_tokenize_words_numbers_and_percentages():
    assert build_indexes.tokenize("GDP is 12.5% (UP 3!)") == ["gdp", "is", "12", "5", "%", "up", "3"]
    assert build_indexes.tokenize("") == []


def test_build_indexes_excludes_duplicates_persists_and_replaces_stale_data(tmp_path, monkeypatch):
    import chromadb

    chunk_file = tmp_path / "chunks.jsonl"
    chroma_path = tmp_path / "chroma"
    monkeypatch.setattr(build_indexes, "chunks_path", lambda _: chunk_file)
    monkeypatch.setattr(build_indexes, "bm25_path", lambda _: tmp_path / "bm25.pkl")
    monkeypatch.setitem(build_indexes.PATHS, "chroma", chroma_path)
    _fake_embeddings(monkeypatch)
    cfg = SimpleNamespace(kb_sig="testkb00", collection_name="kb_testkb00", kb={"embed_model": "test-model"})

    write_jsonl(chunk_file, [
        _chunk("keep__a", "Final exam is 50 percent."),
        _chunk("keep__b", "Attendance limit is 80 percent."),
        _chunk("skip__duplicate", "Repeated text", duplicate_of="keep__a"),
    ])
    assert build_indexes.build_indexes(cfg) == 2

    index = pickle.loads((tmp_path / "bm25.pkl").read_bytes())
    assert set(index) == {"kb_sig", "chunk_ids", "tokenized_corpus", "bm25"}
    assert index["kb_sig"] == "testkb00"
    assert index["chunk_ids"] == ["keep__a", "keep__b"]
    assert len(index["tokenized_corpus"]) == 2
    assert index["bm25"].get_scores(["attendance"]).shape == (2,)

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection("kb_testkb00")
    assert collection.count() == 2
    assert collection.metadata["hnsw:space"] == "cosine"
    stored = collection.get(include=["metadatas", "documents"])
    assert set(stored["ids"]) == {"keep__a", "keep__b"}
    meta = {m["chunk_id"]: m for m in stored["metadatas"]}
    assert meta["keep__a"]["course_code"] == "c01"
    assert meta["keep__a"]["sessions"] == "2;3"
    assert "Session 2" in stored["documents"][0]
    nearest = collection.query(query_embeddings=[[1.0, 0.0]], n_results=2, include=["distances"])
    assert nearest["ids"][0][0] == "keep__a"
    assert nearest["distances"][0] == pytest.approx([0.0, 1.0])

    original = _chunk("keep__a", "Final exam is 50 percent.")
    assert build_indexes.load_chunk_by_id("keep__a", cfg) == original
    assert build_indexes.load_chunk_by_id("keep__a", cfg).to_dict() == original.to_dict()

    # A subsequent build must remove IDs that are no longer in the chunk file.
    write_jsonl(chunk_file, [_chunk("new__only", "Only the new content." )])
    assert build_indexes.build_indexes(cfg) == 1
    rebuilt_client = chromadb.PersistentClient(path=str(chroma_path))
    rebuilt = rebuilt_client.get_collection("kb_testkb00")
    assert rebuilt.count() == 1
    assert rebuilt.get()["ids"] == ["new__only"]
    second_index = pickle.loads((tmp_path / "bm25.pkl").read_bytes())
    assert second_index["chunk_ids"] == ["new__only"]


def test_build_indexes_requires_at_least_one_indexable_chunk(tmp_path, monkeypatch):
    chunk_file = tmp_path / "chunks.jsonl"
    write_jsonl(chunk_file, [_chunk("duplicate", "same", duplicate_of="canonical")])
    monkeypatch.setattr(build_indexes, "chunks_path", lambda _: chunk_file)
    cfg = SimpleNamespace(kb_sig="testkb00", collection_name="kb_testkb00", kb={"embed_model": "test-model"})

    with pytest.raises(ValueError, match="No non-duplicate chunks"):
        build_indexes.build_indexes(cfg)


def test_staging_failures_preserve_previous_working_indexes(tmp_path, monkeypatch):
    import chromadb

    chunk_file = tmp_path / "chunks.jsonl"
    chroma_path = tmp_path / "chroma"
    bm25_file = tmp_path / "bm25.pkl"
    monkeypatch.setattr(build_indexes, "chunks_path", lambda _: chunk_file)
    monkeypatch.setattr(build_indexes, "bm25_path", lambda _: bm25_file)
    monkeypatch.setitem(build_indexes.PATHS, "chroma", chroma_path)
    _fake_embeddings(monkeypatch)
    cfg = SimpleNamespace(kb_sig="testkb00", collection_name="kb_testkb00", kb={"embed_model": "test-model"})

    write_jsonl(chunk_file, [_chunk("old__id", "Final old version.")])
    assert build_indexes.build_indexes(cfg) == 1
    old_bm25 = bm25_file.read_bytes()

    def assert_old_indexes_remain():
        client = chromadb.PersistentClient(path=str(chroma_path))
        assert client.get_collection("kb_testkb00").get()["ids"] == ["old__id"]
        assert bm25_file.read_bytes() == old_bm25

    write_jsonl(chunk_file, [_chunk("new__id", "Final new version.")])
    monkeypatch.setattr(
        build_indexes,
        "_write_bm25_staging",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected BM25 staging failure")),
    )
    with pytest.raises(OSError, match="injected BM25 staging failure"):
        build_indexes.build_indexes(cfg)
    assert_old_indexes_remain()

    monkeypatch.undo()
    # Restore the fixture-specific path and model patches after undoing the injected failure.
    monkeypatch.setattr(build_indexes, "chunks_path", lambda _: chunk_file)
    monkeypatch.setattr(build_indexes, "bm25_path", lambda _: bm25_file)
    monkeypatch.setitem(build_indexes.PATHS, "chroma", chroma_path)
    _fake_embeddings(monkeypatch)
    populate = build_indexes._populate_staging_collection

    def fail_after_staging(client, name, chunks, vectors, model_name):
        populate(client, name, chunks, vectors, model_name)
        raise RuntimeError("injected Chroma staging failure")

    monkeypatch.setattr(build_indexes, "_populate_staging_collection", fail_after_staging)
    with pytest.raises(RuntimeError, match="injected Chroma staging failure"):
        build_indexes.build_indexes(cfg)
    assert_old_indexes_remain()


def test_bm25_promotion_failure_rolls_back_chroma_and_preserves_pickle(tmp_path, monkeypatch):
    import chromadb

    chunk_file = tmp_path / "chunks.jsonl"
    chroma_path = tmp_path / "chroma"
    bm25_file = tmp_path / "bm25.pkl"
    monkeypatch.setattr(build_indexes, "chunks_path", lambda _: chunk_file)
    monkeypatch.setattr(build_indexes, "bm25_path", lambda _: bm25_file)
    monkeypatch.setitem(build_indexes.PATHS, "chroma", chroma_path)
    _fake_embeddings(monkeypatch)
    cfg = SimpleNamespace(kb_sig="testkb00", collection_name="kb_testkb00", kb={"embed_model": "test-model"})

    write_jsonl(chunk_file, [_chunk("old__id", "Final old version.")])
    build_indexes.build_indexes(cfg)
    old_bm25 = bm25_file.read_bytes()
    write_jsonl(chunk_file, [_chunk("new__id", "Final new version.")])

    monkeypatch.setattr(build_indexes.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("injected promote failure")))
    with pytest.raises(RuntimeError, match="previous indexes were restored"):
        build_indexes.build_indexes(cfg)

    client = chromadb.PersistentClient(path=str(chroma_path))
    assert client.get_collection("kb_testkb00").get()["ids"] == ["old__id"]
    assert bm25_file.read_bytes() == old_bm25
