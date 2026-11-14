"""Tests for the readable syllabus tables and the course summary chunks (no real PDFs needed)."""
import json

import pytest

from src.kb import syllabus_text
from src.kb.syllabus_text import clean_cell, course_summary_chunks, section_text, table_to_text

# 1. a piece of the real Management Control evaluation table, as PyMuPDF reads it (no header, glued words)
EVALUATION_TABLE = (
    "| FinalExam | 40% | organizations . Understand transfer pricing - withinorganizations . "
    "| Thefinalexamwillcoverallcoursecontent Aminimumgradeof . 45isrequiredonthefinalexamtopassthesubject . |\n"
    "| --- | --- | --- | --- |\n"
    "| Class Participation | 10% | Understand the different - typesofresponsibilitycenters | |"
)
# 2. the workload table, with the decimal points lost
WORKLOAD_TABLE = (
    "| LearningActivity | Weighting | Estimatedtimeastudentshould dedicatetoprepareforandparticipate in |\n"
    "| --- | --- | --- |\n"
    "| Lectures | 133% . | 200hours . |\n"
    "| TOTAL | 1000% . | 1500hours . |"
)


def test_evaluation_table_becomes_one_line_per_component():
    text = table_to_text(EVALUATION_TABLE)
    lines = text.splitlines()
    assert lines[0].startswith("- Final Exam: 40%.")
    assert lines[1].startswith("- Class Participation: 10%.")
    assert "responsibility" not in text  # the learning-objectives column is dropped
    assert "organizations" not in text


def test_lost_decimals_come_back():
    text = table_to_text(WORKLOAD_TABLE)
    assert "- Lectures: 13.3% of the time, 20.0 hours" in text
    assert "- TOTAL: 100.0% of the time, 150.0 hours" in text
    assert "Estimated" not in text  # the header row is not repeated as a line


def test_glued_words_are_split_when_the_word_list_is_installed():
    pytest.importorskip("wordninja")
    text = table_to_text(EVALUATION_TABLE)
    assert "The final exam will cover all course content" in text
    assert "A minimum grade of 4.5 is required on the final exam" in text


def test_real_long_words_are_not_split():
    pytest.importorskip("wordninja")
    assert clean_cell("Macroeconomics") == "Macroeconomics"
    assert clean_cell("regularization") == "regularization"  # "regular iz ation" is rejected
    assert clean_cell("Discussions") == "Discussions"


def test_section_text_keeps_the_body_and_adds_the_clean_table():
    section = {"text": "EVALUATION CRITERIA body", "tables": [EVALUATION_TABLE]}
    text = section_text(section)
    assert text.startswith("EVALUATION CRITERIA body\n\n- Final Exam: 40%.")
    assert section_text({"text": "no tables", "tables": []}) == "no tables"


def _write_table(path, rows):
    fields = list(rows[0])
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join('"' + str(row[f]).replace('"', '""') + '"' for f in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_course_summary_chunks_from_the_course_table(tmp_path, monkeypatch):
    table = tmp_path / "course_table.csv"
    _write_table(table, [
        {"course_code": "mc", "course_name": "Management Control", "final_exam_weight": "40.0",
         "components": json.dumps([{"name": "Final Exam", "weight": 40.0}, {"name": "Assignments", "weight": 60.0}]),
         "attendance_limit_pct": "80.0", "retake_policy": "Re-sit in June/July.", "professor": "ANA PEREZ",
         "n_sessions": "25", "language": "en", "evidence_section_ids": "mc_syl_sec08;mc_syl_sec10",
         "human_checked": "False"},
        {"course_code": "ma", "course_name": "Macroeconomics", "final_exam_weight": "",
         "components": "[]", "attendance_limit_pct": "", "retake_policy": "", "professor": "",
         "n_sessions": "", "language": "en", "evidence_section_ids": "", "human_checked": "False"},
    ])
    monkeypatch.setitem(syllabus_text.PATHS, "course_table", table)
    monkeypatch.setitem(syllabus_text.PATHS, "syllabus_sections", tmp_path / "no_sections.jsonl")

    chunks, parents = course_summary_chunks("abc123")

    assert [c.chunk_id for c in chunks] == ["course_row__mc", "course_row__ma"]
    first = chunks[0]
    assert first.doc_type == "course_row" and first.course_code == "mc" and first.kb_sig == "abc123"
    assert first.covers == ["mc_syl_sec08", "mc_syl_sec10"]  # the evaluation's gold_ids still match
    assert "The final exam is worth 40% of the final grade." in first.text
    assert "How the course is evaluated: Final Exam 40%, Assignments 60%." in first.text
    assert "Minimum attendance: 80%" in first.text and "Professor: Ana Perez, who teaches this course." in first.text
    assert "There is no final exam in this course." in chunks[1].text  # empty value -> said plainly, not invented
    assert [p.parent_id for p in parents] == ["parent__course_row__mc", "parent__course_row__ma"]


def test_no_course_table_means_no_summary_chunks(tmp_path, monkeypatch):
    monkeypatch.setitem(syllabus_text.PATHS, "course_table", tmp_path / "missing.csv")
    assert course_summary_chunks("abc123") == ([], [])


def test_section_heading_uses_the_real_title():
    from src.kb.syllabus_text import section_heading
    assert section_heading({"section_id": "x_syl_sec03", "section_name": "OTHER", "text": "SUBJECT DESCRIPTION\nThis course..."}) == "Subject Description"
    assert section_heading({"section_id": "x_syl_sec01", "section_name": "OTHER", "text": "MANAGEMENT CONTROL\nDual Degree"}) == "Course details"
    assert section_heading({"section_id": "x_syl_sec10", "section_name": "RESIT", "text": "RE-SIT / RE-TAKE POLICY\nEach..."}) == "Re-sit / Re-take Policy"
    # 1. "Professor: NAME" is not a heading, so the category is used
    assert section_heading({"section_id": "x_syl_sec02", "section_name": "PROFESSOR", "text": "Professor: ANA PEREZ\n..."}) == "Professor"


def test_with_course_puts_the_course_name_first():
    from src.kb.syllabus_text import with_course
    assert with_course("Management Control", "Session 3 › Introduction", "text") == "Management Control › Session 3 › Introduction\ntext"
    assert with_course("", "Session 3", "text") == "Session 3\ntext"


def test_noise_pieces_are_detected():
    from src.kb.syllabus_text import is_noise
    for text in ["THANK YOU", "THANK YOU!\n21", "31", "→\n→\n→", "≈"]:
        assert is_noise(text, has_image_text=False), text
    for text in ["Static budgets", "gg_season()", "Takeaways"]:
        assert not is_noise(text, has_image_text=False), text
    assert not is_noise("31", has_image_text=True)  # the image text can carry the content


def test_course_summary_has_teacher_wording_title_and_books(tmp_path, monkeypatch):
    table = tmp_path / "course_table.csv"
    _write_table(table, [
        {"course_code": "ob", "course_name": "Organizational Behaviour", "final_exam_weight": "40.0",
         "components": "[]", "attendance_limit_pct": "", "retake_policy": "", "professor": "CARMEN MORALES",
         "n_sessions": "", "language": "en", "evidence_section_ids": "ob_syl_sec08", "human_checked": "False"},
    ])
    sections = tmp_path / "syllabus_sections.jsonl"
    sections.write_text(
        json.dumps({"section_id": "ob_syl_sec01", "source_id": "ob_syl", "course_code": "ob", "section_name": "OTHER",
                    "text": "ORGANIZATIONAL BEHAVIOR\nDual Degree in ...", "tables": []}) + "\n"
        + json.dumps({"section_id": "ob_syl_sec10", "source_id": "ob_syl", "course_code": "ob", "section_name": "BIBLIOGRAPHY",
                      "text": "BIBLIOGRAPHY\nCompulsory\n- Robbins. Organizational Behavior. null", "tables": []}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(syllabus_text.PATHS, "course_table", table)
    monkeypatch.setitem(syllabus_text.PATHS, "syllabus_sections", sections)

    chunk = course_summary_chunks("abc")[0][0]

    assert "syllabus title: Organizational Behavior" in chunk.text  # the other spelling is searchable too
    assert "Professor: Carmen Morales, who teaches this course." in chunk.text  # answers "who teaches...?"
    assert "Bibliography (the book list for this course): Compulsory - Robbins. Organizational Behavior." in chunk.text
    assert "null" not in chunk.text
    assert chunk.covers == ["ob_syl_sec08", "ob_syl_sec01", "ob_syl_sec10"]


def test_chunks_are_cut_at_sentence_ends_not_inside_words():
    from src.kb.syllabus_text import split_into_chunks
    # 1. the PDF breaks sentences at the end of each printed line: they are joined back before cutting
    text = "The course spans the full ML\nworkflow from data curation. It ends with deep\nlearning. Reinforcement learning is introduced."
    chunks = split_into_chunks(text, 60, 0)
    assert chunks == ["The course spans the full ML workflow from data curation.",
                      "It ends with deep learning.",
                      "Reinforcement learning is introduced."]
    for chunk in split_into_chunks("word " * 300, 100, 20):  # one giant sentence: cut at spaces
        assert len(chunk) <= 100 and not chunk.startswith(" ")


def test_overlap_repeats_the_last_sentence():
    from src.kb.syllabus_text import split_into_chunks
    chunks = split_into_chunks("One two three. Four five six. Seven eight nine.", 35, 15)
    assert chunks == ["One two three. Four five six.", "Four five six. Seven eight nine."]  # carried over


def test_table_lines_stay_separate_units():
    from src.kb.syllabus_text import split_into_chunks
    text = "Body text.\n\n- Final Exam: 40%. Comment.\n- Class Participation: 10%."
    # 2. a component and its comment stay together in the same chunk
    assert split_into_chunks(text, 30, 0) == ["Body text.", "- Final Exam: 40%. Comment.", "- Class Participation: 10%."]


def test_programme_chunks_hold_whole_sessions_and_know_their_numbers():
    from src.kb.syllabus_text import session_numbers, split_into_chunks
    text = "PROGRAM SESSION 1 (LIVE) Topic one. SESSION 2 (LIVE) Topic two. SESSIONS 9-10-11 Case. SESSIONS 29 - 30 Review."
    chunks = split_into_chunks(text, 40, 20, by_session=True)
    assert chunks == ["PROGRAM SESSION 1 (LIVE) Topic one.", "SESSION 2 (LIVE) Topic two.",
                      "SESSIONS 9-10-11 Case.", "SESSIONS 29 - 30 Review."]
    assert session_numbers("SESSIONS 9-10-11 Case.", []) == [9, 10, 11]
    assert session_numbers("SESSIONS 29 - 30 Review.", []) == [29, 30]
    assert session_numbers("continues the case study", [4, 5]) == [5]  # a continuation keeps the last session


def test_broken_hyphens_are_joined():
    from src.kb.syllabus_text import fix_broken_hyphens
    assert fix_broken_hyphens("exams (re-\nsits) and (re- sits)") == "exams (re-sits) and (re-sits)"
    assert fix_broken_hyphens("short- and long-term") == "short- and long-term"


def test_cross_course_rows_say_when_there_is_no_final_exam():
    from src.generation.table_qa import _row_text
    from src.schemas import CourseRow
    row = CourseRow(course_code="ma", course_name="Macroeconomics", final_exam_weight=None,
                    components='[{"name": "Intermediate Tests", "weight": 60.0}]', attendance_limit_pct=None,
                    retake_policy="", professor="", n_sessions=None, language="en", evidence_section_ids="")
    text = _row_text(row)
    assert "no final exam" in text and "Intermediate Tests: 60%" in text
