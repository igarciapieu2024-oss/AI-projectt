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
    cfg = load_config("configs/baselines/v1_2_small_model.yaml")
    content = build_messages("What is the final exam worth?", [_hit()], cfg)[-1]["content"]
    assert content.rstrip().endswith("Answer in 1 to 3 sentences, with only what the question asks.")
    assert "say that the material does not cover it." in content
    assert "in square brackets" in content


def test_v1_prompt_is_unchanged():
    cfg = load_config("configs/baselines/b1_first_version.yaml")
    content = build_messages("What is the final exam worth?", [_hit()], cfg)[-1]["content"]
    assert "in square brackets" not in content  # the baseline keeps its original prompt


def test_v3_prompt_names_the_course_of_each_source_and_has_the_rules():
    # 1. the source header starts with the course name, taken from the searched text
    hit = Hit(
        chunk=Chunk(chunk_id="slide__macro_s01_p016__00", kb_sig="t", strategy="slide", doc_type="slide",
                    course_code="macro", sessions=[1], source_id="macro_s01", covers=["macro_s01_p016"],
                    parent_id="macro_s01_p016", outline_path="Session 1 › Midterm",
                    text="Midterm Exam: Session 16", embed_text="Macroeconomics › Session 1 › Midterm\nMidterm Exam: Session 16"),
        rank=1, label="S1",
    )
    cfg = load_config("configs/experiments/01_prompt_rules.yaml")
    messages = build_messages("When is the midterm?", [hit], cfg)
    assert "[S1] Macroeconomics › Session 1 › Midterm" in messages[-1]["content"]
    assert "answer for each course separately" in messages[0]["content"]
    assert messages[-1]["content"].rstrip().endswith("with only what the question asks.")
    # 2. v2 keeps the old header, so the evaluated version does not change
    v2 = build_messages("When is the midterm?", [hit], load_config("configs/baselines/v1_2_small_model.yaml"))
    assert "[S1] Session 1 › Midterm" in v2[-1]["content"]
