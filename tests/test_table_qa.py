import csv

from src.generation import table_qa


def _write_table(path):
    fields = [
        "course_code", "course_name", "final_exam_weight", "components",
        "attendance_limit_pct", "retake_policy", "professor", "n_sessions",
        "language", "evidence_section_ids", "human_checked",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "course_code": "c01",
            "course_name": "Management Control",
            "final_exam_weight": "40",
            "components": '[{"name":"Final exam","weight":40},{"name":"Project","weight":60}]',
            "attendance_limit_pct": "20",
            "retake_policy": "",
            "professor": "",
            "n_sessions": "",
            "language": "en",
            "evidence_section_ids": "c01_syl_sec01;c01_syl_sec02",
            "human_checked": "true",
        })
        writer.writerow({
            "course_code": "c02",
            "course_name": "Macroeconomics",
            "final_exam_weight": "",
            "components": "",
            "attendance_limit_pct": "",
            "retake_policy": "One retake",
            "professor": "Professor A",
            "n_sessions": "12",
            "language": "es",
            "evidence_section_ids": "c02_syl_sec01",
            "human_checked": "false",
        })


def test_missing_course_table_returns_no_hits(monkeypatch, tmp_path):
    monkeypatch.setitem(table_qa.PATHS, "course_table", tmp_path / "missing.csv")

    assert table_qa.answer_from_table("Compare courses", object()) == []


def test_course_rows_become_hits_and_named_course_is_selected(monkeypatch, tmp_path):
    path = tmp_path / "course_table.csv"
    _write_table(path)
    monkeypatch.setitem(table_qa.PATHS, "course_table", path)
    cfg = type("Config", (), {"kb_sig": "test"})()

    hits = table_qa.answer_from_table("Management Control final exam", cfg)

    assert len(hits) == 1
    assert hits[0].label == "S1"
    assert hits[0].chunk.doc_type == "course_row"
    assert hits[0].chunk.course_code == "c01"
    assert hits[0].chunk.covers == ["c01_syl_sec01", "c01_syl_sec02"]
    assert "final exam 40%" in hits[0].chunk.text
    assert "Project: 60%" in hits[0].chunk.text
    assert "attendance limit 20%" in hits[0].chunk.text

    all_hits = table_qa.answer_from_table("Compare my courses", cfg)
    assert [hit.chunk.course_code for hit in all_hits] == ["c01", "c02"]
    assert "no final exam" in all_hits[1].chunk.text  # said explicitly, never a made-up weight
    assert "retake: One retake" in all_hits[1].chunk.text


def test_comparison_questions_get_the_rows_in_order():
    from src.generation.table_qa import _ordered_for
    rows = [
        {"course_code": "mc", "final_exam_weight": "40.0", "n_sessions": "25"},
        {"course_code": "ma", "final_exam_weight": "", "n_sessions": "30"},
        {"course_code": "st", "final_exam_weight": "45.0", "n_sessions": "30"},
        {"course_code": "os", "final_exam_weight": "20.0", "n_sessions": "30"},
    ]
    codes = lambda q: [r["course_code"] for r in _ordered_for(q, rows)]
    assert codes("Which courses have the highest final exam weight?") == ["st", "mc", "os", "ma"]
    assert codes("Which course has the lowest final exam weight?") == ["os", "mc", "st", "ma"]  # no value last
    assert codes("Which course has the fewest sessions?")[0] == "mc"
    assert codes("Tell me about my courses") == ["mc", "ma", "st", "os"]  # nothing to compare: order kept


def test_comparison_questions_start_with_the_ranking_written_out():
    from src.generation.table_qa import _ranking_text
    rows = [
        {"course_name": "Strategies", "final_exam_weight": "45.0", "n_sessions": "20"},
        {"course_name": "Management Control", "final_exam_weight": "40.0", "n_sessions": "20"},
        {"course_name": "Organizational Behaviour", "final_exam_weight": "40.0", "n_sessions": "25"},
        {"course_name": "Macroeconomics", "final_exam_weight": "", "n_sessions": "30"},
    ]
    # 1. the top course stands alone in the first sentence; ties are grouped, never merged with the top
    assert _ranking_text("Which courses have the highest final exam weight?", rows) == (
        "Highest final exam weight: Strategies (45%)."
        " Then: Management Control and Organizational Behaviour (40%). Macroeconomics: no final exam."
    )
    assert _ranking_text("Which course has the fewest sessions?", rows).startswith(
        "Lowest number of sessions: Strategies and Management Control (20)."
    )
    assert _ranking_text("Tell me about my courses", rows) is None  # nothing compared: no ranking
    assert _ranking_text("What is the final exam worth?", rows[:1]) is None  # one course: nothing to rank
