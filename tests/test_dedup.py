import sys
import types
from types import SimpleNamespace

import numpy as np

from src.kb import dedup
from src.schemas import Chunk
from src.utils.io import read_jsonl, write_jsonl


def _chunk(chunk_id, strategy="slide", doc_type="slide", duplicate_of=None):
    return Chunk(
        chunk_id=chunk_id, kb_sig="testkb00", strategy=strategy, doc_type=doc_type,
        course_code="c01", sessions=[1], source_id="c01_s01", covers=["slide1"],
        parent_id="parent1", outline_path="Session 1", text=chunk_id,
        embed_text=chunk_id, duplicate_of=duplicate_of,
    )


def test_dedup_marks_cluster_leaves_noise_and_syllabus_unchanged(tmp_path, monkeypatch):
    source = tmp_path / "chunks.jsonl"
    report = tmp_path / "dedup_report__testkb00.csv"
    rows = [
        _chunk("slide__b"), _chunk("slide__a"),
        _chunk("slide__noise1"), _chunk("slide__noise2"),
        _chunk("section__a", "section", "syllabus_section", "existing"),
    ]
    write_jsonl(source, rows)
    monkeypatch.setattr(dedup, "chunks_path", lambda _: source)
    monkeypatch.setitem(dedup.PATHS, "reports", tmp_path)

    vectors = np.array([[1.0, 0.0], [0.999, 0.01], [0.0, 1.0], [-1.0, 0.0]])

    class FakeModel:
        def __init__(self, name):
            assert name == "test-model"

        def encode(self, texts, **kwargs):
            assert texts == [row.embed_text for row in rows[:4]]
            return vectors

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeModel))
    cfg = SimpleNamespace(kb_sig="testkb00", kb={"embed_model": "test-model"})

    assert dedup.dedup(cfg) == 1
    output = read_jsonl(source, Chunk)
    by_id = {chunk.chunk_id: chunk for chunk in output}
    assert by_id["slide__a"].duplicate_of is None
    assert by_id["slide__b"].duplicate_of == "slide__a"
    assert by_id["slide__noise1"].duplicate_of is None
    assert by_id["slide__noise2"].duplicate_of is None
    assert by_id["section__a"].duplicate_of == "existing"
    report_lines = report.read_text(encoding="utf-8").splitlines()
    assert report_lines[0] == "cluster_id,representative_id,duplicate_id,cosine_distance"
    assert "slide__a,slide__b" in report_lines[1]


def test_dedup_with_fewer_than_two_slide_chunks_does_not_load_model(tmp_path, monkeypatch):
    source = tmp_path / "chunks.jsonl"
    write_jsonl(source, [_chunk("slide__one")])
    monkeypatch.setattr(dedup, "chunks_path", lambda _: source)
    monkeypatch.setitem(dedup.PATHS, "reports", tmp_path)
    cfg = SimpleNamespace(kb_sig="testkb00", kb={"embed_model": "test-model"})

    assert dedup.dedup(cfg) == 0
    assert (tmp_path / "dedup_report__testkb00.csv").exists()


def test_dedup_chaining_only_marks_members_close_to_representative(tmp_path, monkeypatch):
    source = tmp_path / "chunks.jsonl"
    rows = [_chunk("slide__c"), _chunk("slide__b"), _chunk("slide__a")]
    write_jsonl(source, rows)
    monkeypatch.setattr(dedup, "chunks_path", lambda _: source)
    monkeypatch.setitem(dedup.PATHS, "reports", tmp_path)

    angles = np.deg2rad([24.0, 12.0, 0.0])
    vectors = np.column_stack((np.cos(angles), np.sin(angles)))

    class FakeModel:
        def __init__(self, name):
            assert name == "test-model"

        def encode(self, texts, **kwargs):
            return vectors

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeModel))
    cfg = SimpleNamespace(kb_sig="testkb00", kb={"embed_model": "test-model"})

    assert dedup.dedup(cfg) == 1
    output = read_jsonl(source, Chunk)
    by_id = {chunk.chunk_id: chunk for chunk in output}
    assert by_id["slide__b"].duplicate_of == "slide__a"
    assert by_id["slide__c"].duplicate_of is None

    report_lines = (tmp_path / "dedup_report__testkb00.csv").read_text(encoding="utf-8").splitlines()
    assert len(report_lines) == 2
    distance = float(report_lines[1].split(",")[-1])
    assert distance <= 0.05


def test_dedup_rejects_invalid_parameters():
    cfg = SimpleNamespace(kb_sig="testkb00", kb={"dedup_eps": 0})
    try:
        dedup.dedup(cfg)
    except ValueError as exc:
        assert "dedup_eps" in str(exc)
    else:
        raise AssertionError("invalid eps should fail before reading chunks")
