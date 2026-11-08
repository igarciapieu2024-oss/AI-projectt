"""Shared formats: every file that exists matches its record class in src/schemas.py."""
import csv
import json

import pytest

from src.config import PATHS, load_config
from src.generation import fake
from src.schemas import CONTRACT_FILES, AnswerResult
from src.utils import ids


@pytest.mark.parametrize("rel_path,record", sorted(CONTRACT_FILES.items()))
def test_file_matches_contract(rel_path, record):
    path = PATHS["root"] / rel_path
    if not path.exists():
        pytest.skip(f"{rel_path} not produced yet")
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            header = next(csv.reader(f))
        assert header == record.field_names(), f"{rel_path}: columns differ from {record.__name__}"
    else:
        with path.open(encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if line.strip():
                    data = json.loads(line)
                    assert set(data) == set(record.field_names()), f"{rel_path} line {n}: fields differ"
                    record.from_dict(data)


def test_fake_answer_matches_contract():
    result = fake.answer("What is the final exam worth?")
    assert isinstance(result, AnswerResult)
    again = AnswerResult.from_dict(json.loads(json.dumps(result.to_dict())))
    assert again == result
    assert [h.label for h in result.context_hits] == ["S1", "S2"]
    assert "total" in result.timings_ms


def test_fake_stream_ends_with_result():
    pieces = list(fake.answer_stream("hello"))
    assert isinstance(pieces[-1], AnswerResult)
    assert all(isinstance(p, str) for p in pieces[:-1])


def test_configs_load_and_share_index():
    base = load_config("configs/base.yaml")
    final = load_config("configs/final.yaml")
    b0 = load_config("configs/b0_no_rag.yaml")
    b1 = load_config("configs/b1_v1.yaml")
    assert final.kb_sig == base.kb_sig          # same knowledge base -> same index
    assert b0.kb_sig == base.kb_sig             # b0 only changes generation
    assert b1.kb_sig != base.kb_sig             # b1 changes the knowledge base
    assert b0.generation["rag"] is False


def test_ids():
    src_id = ids.source_id("c01", "slides", [5, 6])
    assert src_id == "c01_s05-06"
    assert ids.slide_id(src_id, 17) == "c01_s05-06_p017"
    assert ids.image_id("c01_s04_p025", 1) == "c01_s04_p025_img01"
    assert ids.image_id("c01_s04_p025") == "c01_s04_p025_full"
    assert ids.section_id("c01_syl", 4) == "c01_syl_sec04"
