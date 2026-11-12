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
    assert "final exam" not in all_hits[1].chunk.text
    assert "retake: One retake" in all_hits[1].chunk.text
