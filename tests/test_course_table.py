import csv
import json

from src.config import PATHS
from src.kb.course_table import build_course_table
from src.schemas import CourseRow, SyllabusSection
from src.utils.io import read_csv, write_jsonl


def _section(
    course_code,
    section_id,
    section_name,
    *,
    text="",
    tables=None,
):
    return SyllabusSection(
        section_id=section_id,
        source_id=f"{course_code}_syl",
        course_code=course_code,
        section_name=section_name,
        page_start=1,
        page_end=1,
        text=text,
        tables=tables or [],
    )


def _write_manifest(path, courses):
    fields = ["source_id", "course_code", "course_name", "doc_type", "sessions", "file", "language"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for code, name in courses:
            writer.writerow({
                "source_id": f"{code}_syl",
                "course_code": code,
                "course_name": name,
                "doc_type": "syllabus",
                "sessions": "",
                "file": f"{code}.pdf",
                "language": "en",
            })


def test_build_course_table_parses_split_components_and_metadata(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    sections_path = tmp_path / "syllabus_sections.jsonl"
    table_path = tmp_path / "course_table.csv"
    _write_manifest(manifest_path, [("mgmt_control", "Management Control")])
    write_jsonl(
        sections_path,
        [
            _section(
                "mgmt_control",
                "mgmt_control_syl_sec06",
                "EVALUATION",
                tables=[
                    "| Criteria | Percentage |\n| --- | --- |\n"
                    "| FinalExam | 40% |\n"
                    "| Intermediatetests | 30% |\n"
                    "| Participation | 10% |"
                ],
            ),
            _section(
                "mgmt_control",
                "mgmt_control_syl_sec09",
                "OTHER",
                text="Assignments\n20 %",
            ),
            _section(
                "mgmt_control",
                "mgmt_control_syl_sec10",
                "RESIT",
                text="80% attendance rule. Students who fail may take one resit exam.",
            ),
            _section(
                "mgmt_control",
                "mgmt_control_syl_sec11",
                "PROFESSOR",
                text="Professor: Taylor Example (taylor@example.edu)",
            ),
            _section(
                "mgmt_control",
                "mgmt_control_syl_sec12",
                "PROGRAM",
                text="SESSION 1\nSESSIONS 24 - 25",
            ),
        ],
    )
    monkeypatch.setitem(PATHS, "manifest", manifest_path)
    monkeypatch.setitem(PATHS, "syllabus_sections", sections_path)
    monkeypatch.setitem(PATHS, "course_table", table_path)

    assert build_course_table(None) == 1

    [row] = read_csv(table_path)
    components = json.loads(row["components"])
    assert row["course_name"] == "Management Control"
    assert row["final_exam_weight"] == "40.0"
    assert [(item["name"], item["weight"]) for item in components] == [
        ("Final Exam", 40.0),
        ("Intermediate Tests", 30.0),
        ("Participation", 10.0),
        ("Assignments", 20.0),
    ]
    assert sum(item["weight"] for item in components) == 100
    assert row["attendance_limit_pct"] == "80.0"
    assert row["retake_policy"] == "80% attendance rule. Students who fail may take one resit exam."
    assert row["professor"] == "Taylor Example"
    assert row["n_sessions"] == "25"
    assert row["evidence_section_ids"] == (
        "mgmt_control_syl_sec06;mgmt_control_syl_sec09;mgmt_control_syl_sec10;"
        "mgmt_control_syl_sec11;mgmt_control_syl_sec12"
    )
    assert list(row) == CourseRow.field_names()


def test_text_adds_missing_evaluation_component_without_duplicate_final(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    sections_path = tmp_path / "syllabus_sections.jsonl"
    table_path = tmp_path / "course_table.csv"
    _write_manifest(manifest_path, [("strategies", "Strategies")])
    write_jsonl(
        sections_path,
        [
            _section(
                "strategies",
                "strategies_syl_sec05",
                "EVALUATION",
                text=(
                    "A. FINAL EXAM (45%) … "
                    "B. ONGOING EVALUATION QUIZZES (25%) … "
                    "C. CASE STUDY COMPETITION (15%) … "
                    "D. PARTICIPATION (15%)"
                ),
                tables=[
                    "| Criteria | Percentage |\n| --- | --- |\n"
                    "| Final Exam | 45% |\n| Ongoing Evaluation Quizzes | 25% |\n"
                    "| Case Study Competition | 15% |"
                ],
            )
        ],
    )
    monkeypatch.setitem(PATHS, "manifest", manifest_path)
    monkeypatch.setitem(PATHS, "syllabus_sections", sections_path)
    monkeypatch.setitem(PATHS, "course_table", table_path)

    assert build_course_table(None) == 1

    [row] = read_csv(table_path)
    components = json.loads(row["components"])
    assert [(item["name"], item["weight"]) for item in components] == [
        ("Final Exam", 45.0),
        ("Ongoing Evaluation Quizzes", 25.0),
        ("Case Study Competition", 15.0),
        ("Participation", 15.0),
    ]
    assert row["final_exam_weight"] == "45.0"
    assert row["evidence_section_ids"] == "strategies_syl_sec05"


def test_text_only_components_and_missing_values_are_not_invented(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    sections_path = tmp_path / "syllabus_sections.jsonl"
    table_path = tmp_path / "course_table.csv"
    _write_manifest(
        manifest_path,
        [("macro", "Macroeconomics"), ("empty", "No Details")],
    )
    write_jsonl(
        sections_path,
        [
            _section(
                "macro",
                "macro_syl_sec04",
                "EVALUATION",
                text="Midterm Exam 60%; Assignments 40%",
            )
        ],
    )
    monkeypatch.setitem(PATHS, "manifest", manifest_path)
    monkeypatch.setitem(PATHS, "syllabus_sections", sections_path)
    monkeypatch.setitem(PATHS, "course_table", table_path)

    assert build_course_table(None) == 2

    rows = {row["course_code"]: row for row in read_csv(table_path)}
    macro = rows["macro"]
    assert macro["final_exam_weight"] == ""
    assert [(item["name"], item["weight"]) for item in json.loads(macro["components"])] == [
        ("Midterm", 60.0),
        ("Assignments", 40.0),
    ]
    empty = rows["empty"]
    assert empty["final_exam_weight"] == ""
    assert json.loads(empty["components"]) == []
    assert empty["attendance_limit_pct"] == ""
    assert empty["retake_policy"] == ""
    assert empty["professor"] == ""
    assert empty["n_sessions"] == ""
    assert empty["evidence_section_ids"] == ""


def test_components_sum_to_100_for_each_expected_course(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    sections_path = tmp_path / "syllabus_sections.jsonl"
    table_path = tmp_path / "course_table.csv"
    courses = [
        ("ai_ml", "AI/ML"),
        ("macro", "Macro"),
        ("mgmt_control", "Management Control"),
        ("os", "Operating Systems"),
        ("ob", "Organizational Behaviour"),
        ("strategies", "Strategies"),
        ("time_series", "Time Series"),
    ]
    _write_manifest(manifest_path, courses)
    expected = {
        "ai_ml": [("Final Exam", 35), ("Group Presentation", 20), ("Individual Work", 30), ("Class Participation", 15)],
        "macro": [("Group Work", 30), ("Class Participation", 10), ("Intermediate Tests", 60)],
        "mgmt_control": [("Final Exam", 40), ("Class Participation", 10), ("Intermediate Tests", 30), ("Assignments", 20)],
        "os": [("Class Participation", 10), ("Individual Work", 20), ("Group Presentation", 30), ("Intermediate Tests", 20), ("Final Exam", 20)],
        "ob": [("Class Participation", 20), ("Group Presentation", 20), ("Midterm", 20), ("Final Exam", 40)],
        "strategies": [("Final Exam", 45), ("Ongoing Evaluation Quizzes", 25), ("Case Study Competition", 15), ("Participation", 15)],
        "time_series": [("Final Exam", 30), ("Group Work", 20), ("Class Participation", 10), ("Intermediate Tests", 30), ("Individual Work", 10)],
    }

    def markdown_table(items):
        return (
            "| Component | Percentage |\n| --- | --- |\n"
            + "\n".join(f"| {name} | {weight}% |" for name, weight in items)
        )

    sections = []
    for code, items in expected.items():
        if code == "strategies":
            text = " … ".join(f"{name} ({weight}%)" for name, weight in items)
            table_items = items[:-1]
        else:
            text = ""
            table_items = items
        sections.append(
            _section(
                code,
                f"{code}_syl_sec01",
                "EVALUATION",
                text=text,
                tables=[markdown_table(table_items)],
            )
        )
    sections.append(
        _section(
            "mgmt_control",
            "mgmt_control_syl_sec02",
            "OTHER",
            text="Assignments\n20 %",
        )
    )
    write_jsonl(sections_path, sections)
    monkeypatch.setitem(PATHS, "manifest", manifest_path)
    monkeypatch.setitem(PATHS, "syllabus_sections", sections_path)
    monkeypatch.setitem(PATHS, "course_table", table_path)

    assert build_course_table(None) == 7

    rows = {row["course_code"]: row for row in read_csv(table_path)}
    for course_code, expected_components in expected.items():
        actual = json.loads(rows[course_code]["components"])
        assert [(item["name"], item["weight"]) for item in actual] == [
            (name, float(weight)) for name, weight in expected_components
        ]
        assert sum(item["weight"] for item in actual) == 100


def test_retake_policy_uses_first_full_sentences_and_skips_title():
    from src.kb.course_table import _retake_policy

    section = _section(
        "os",
        "os_syl_sec03",
        "RESIT",
        text=(
            "RE-SIT / RE-TAKE POLICY\n"
            "Students may resit the exam in June. The retake exam covers all course material. "
            "The third call is available under university rules. A later unrelated sentence."
        ),
    )

    policy, section_id = _retake_policy([section])

    assert policy == (
        "Students may resit the exam in June. The retake exam covers all course material. "
        "The third call is available under university rules."
    )
    assert section_id == section.section_id


def test_non_component_narrative_weight_is_rejected():
    from src.kb.course_table import _text_components

    assert _text_components(
        "Personal / Work Reasons The Final Exam Will Have A Weight Of 60%",
        require_component_term=True,
    ) == []
