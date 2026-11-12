"""The v2 prompt ends with the citation rule, so the small local model keeps the [S1] labels."""
from src.config import load_config
from src.generation.prompts import build_messages
from src.schemas import Chunk, Hit


def _hit():
    chunk = Chunk(chunk_id="c1", kb_sig="x", strategy="section", doc_type="course_row", course_code="mc",
                  sessions=[], source_id="mc_syl", covers=["mc_syl_sec08"], parent_id="p1",
                  outline_path="Course summary › Management Control", text="The final exam is worth 40%.",
                  embed_text="The final exam is worth 40%.")
    return Hit(chunk=chunk, rank=1, label="S1")


def test_v2_prompt_ends_with_the_citation_rule():
    cfg = load_config("configs/final.yaml")
    content = build_messages("What is the final exam worth?", [_hit()], cfg)[-1]["content"]
    assert content.rstrip().endswith("say that the material does not cover it.")
    assert "in square brackets" in content


def test_v1_prompt_is_unchanged():
    cfg = load_config("configs/b1_v1.yaml")
    content = build_messages("What is the final exam worth?", [_hit()], cfg)[-1]["content"]
    assert "in square brackets" not in content  # the baseline keeps its original prompt
