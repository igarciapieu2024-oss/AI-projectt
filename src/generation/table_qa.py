# Answer cross-course questions from the course table (data/processed/course_table.csv, built by
# src/kb/course_table.py). Called by _prepare_answer() in src/generation/answer.py when the router labels a
# question cross_course: instead of searching chunks of 7 syllabi at once, every course becomes one source
# row with its facts, and for questions that rank courses the answer is written out directly.
# course_rows() and asks_for_ranking() are also used by answer.py for the course filter and the per-course answers.
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from src.config import PATHS
from src.retrieval.filters import mentioned_course_codes
from src.schemas import Chunk, CourseRow, Hit
from src.utils.io import read_csv


# The key of one version of the file (path, modification time, size), so a rebuilt table is read again.
def _file_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


# Read the course table once per version and keep it in memory.
@lru_cache(maxsize=2)
def _load_rows(path: str, modified_ns: int, size: int) -> tuple[dict[str, str], ...]:
    # The time and size are only part of the cache key.
    del modified_ns, size
    # A tuple, so the cached rows cannot be changed by mistake.
    return tuple(read_csv(path))


# The evaluation components of a row, stored as JSON text in the CSV, as a list of {"name", "weight"}.
def _components(value: str) -> list[dict]:
    if not value:
        return []
    try:
        components = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"course table has invalid components JSON: {value!r}") from exc
    if not isinstance(components, list) or any(not isinstance(item, dict) for item in components):
        raise ValueError("course table components must be a JSON list of objects")
    return components


# A number from the CSV, or None when the cell is empty (for example, a course without a final exam).
def _optional_float(value: str) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"course table contains an invalid numeric value: {value!r}") from exc


# A whole number from the CSV (the number of sessions), or None when the cell is empty.
def _optional_int(value: str) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"course table contains an invalid session count: {value!r}") from exc


# Write one course row as one line of text, the source text the LLM reads and cites.
def _row_text(row: CourseRow) -> str:
    parts = [f"{row.course_name} ({row.course_code})"]
    if row.final_exam_weight is not None:
        parts.append(f"final exam {row.final_exam_weight:g}%")
    else:
        # Said explicitly: without it the model read another component as the final exam.
        parts.append("no final exam")
    components = _components(row.components)
    if components:
        formatted = ", ".join(
            f"{item.get('name', 'component')}: {item['weight']:g}%"
            for item in components
            if isinstance(item.get("weight"), (int, float))
        )
        if formatted:
            parts.append(f"components: {formatted}")
    if row.attendance_limit_pct is not None:
        parts.append(f"attendance limit {row.attendance_limit_pct:g}%")
    if row.retake_policy:
        parts.append(f"retake: {row.retake_policy}")
    if row.professor:
        parts.append(f"professor: {row.professor}")
    if row.n_sessions is not None:
        parts.append(f"{row.n_sessions} sessions")
    if row.language:
        parts.append(f"language: {row.language}")
    return "; ".join(parts)


# Which column of the table a comparison question is about, found from its words.
_SORT_FIELDS = [
    (re.compile(r"final[\s-]exam", re.I), "final_exam_weight"),
    (re.compile(r"attendance|absence|absences|miss", re.I), "attendance_limit_pct"),
    (re.compile(r"sessions|classes|long", re.I), "n_sessions"),
]
# Words that ask for the bottom of the ranking instead of the top.
_ASCENDING = re.compile(r"\b(lowest|least|smallest|fewest|minimum|less|shortest)\b", re.I)
# Words that ask to order or compare courses. Without one of them the question only mentions the field but
# asks something else (a minimum grade, what happens below the attendance limit), and the ranking line
# would answer a different question.
_RANKING = re.compile(
    r"\b(highest|lowest|most|least|largest|smallest|larger|smaller|higher|lower|bigger|biggest|greatest|"
    r"fewest|longest|shortest|compare|comparison|rank|ranking|order)\b",
    re.I,
)


# Words that ask for something the ranking line does not hold (who teaches, a book, a date...). A question
# that ranks a field but asks for one of these is answered by the model from the course rows instead.
_OTHER_ASK = re.compile(
    r"\b(who|whom|professor|lecturer|teach\w*|book|textbook|bibliography|when|where|which session|date)\b",
    re.I,
)

# How each compared column is named in the ranking line, its unit, and what is written for a course
# without a value.
_FIELD_WORDS = {
    "final_exam_weight": ("final exam weight", "%", "no final exam"),
    "attendance_limit_pct": ("minimum attendance", "%", "not stated in the syllabus"),
    "n_sessions": ("number of sessions", "", "not stated in the syllabus"),
}


# The column the question compares, or None if it compares none of them.
def _compared_field(question: str) -> str | None:
    return next((name for pattern, name in _SORT_FIELDS if pattern.search(question)), None)


# The answer to a ranking question written out as one line: the top course(s) first, then the rest in order,
# with courses of equal value grouped together. Sorting the rows and letting the model read them was not
# enough: the small local model skipped the first row or merged rows with different values. Returns None when
# the question is not a ranking question. The rows arrive already sorted by _ordered_for().
def _ranking_text(question: str, rows: list[dict[str, str]]) -> str | None:
    field = _compared_field(question)
    if field is None or len(rows) < 2 or not _RANKING.search(question) or _OTHER_ASK.search(question):
        return None
    label, unit, missing = _FIELD_WORDS[field]
    lowest = bool(_ASCENDING.search(question))
    # Groups of courses with the same value, in order; courses without a value are listed apart.
    groups: list[tuple[float, list[str]]] = []
    without = []
    for row in rows:
        try:
            value = float(row.get(field) or "")
        except ValueError:
            without.append(row["course_name"])
            continue
        if groups and groups[-1][0] == value:
            groups[-1][1].append(row["course_name"])
        else:
            groups.append((value, [row["course_name"]]))
    if not groups:
        return None

    # Two names joined with "and", more with commas.
    def names(group):
        return " and ".join(group[1]) if len(group[1]) == 2 else ", ".join(group[1])

    top = groups[0]
    text = f"{'Lowest' if lowest else 'Highest'} {label}: {names(top)} ({top[0]:g}{unit})."
    if len(groups) > 1:
        text += " Then: " + "; ".join(f"{names(g)} ({g[0]:g}{unit})" for g in groups[1:]) + "."
    if without:
        text += f" {', '.join(without)}: {missing}."
    return text


# Sort the course rows by the column the question compares (highest first, or lowest first when the
# question asks for the lowest), so the model does not have to compare 7 rows itself, which it got wrong.
# Rows without a value go last. Questions that compare no column keep the table order.
def _ordered_for(question: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    field = _compared_field(question)
    if field is None:
        return rows
    ascending = bool(_ASCENDING.search(question))

    def key(row):
        try:
            value = float(row.get(field) or "")
        except ValueError:
            return (1, 0.0)  # no value: at the end
        return (0, value if ascending else -value)

    return sorted(rows, key=key)


# Return the sources for a cross-course question: one Hit per course (only the courses the question names,
# or all of them when it names none), sorted for the question, with the ranking line first when it applies.
# These Hits replace the search results; their covers are the syllabus sections the facts came from, so
# the evaluation can check them like any other source.
def answer_from_table(question: str, cfg) -> list[Hit]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be empty")
    path = PATHS["course_table"]
    # Without the table there are no table sources; answer.py then searches normally.
    if not path.is_file():
        return []
    rows = list(_load_rows(*_file_key(path)))
    named_codes = set(mentioned_course_codes(question, rows))
    selected = [row for row in rows if not named_codes or row["course_code"] in named_codes]
    selected = _ordered_for(question, selected)
    hits = []
    ranking = _ranking_text(question, selected)
    if ranking:
        # S1 is the ranking already written out; the course rows follow as S2, S3... for details and citations.
        # answer.py returns this line as the answer without calling the model (_ranking_answer()).
        covers = sorted({value for row in selected for value in row.get("evidence_section_ids", "").split(";") if value})
        chunk = Chunk(
            chunk_id="course_ranking", kb_sig=cfg.kb_sig, strategy="section", doc_type="course_row",
            course_code="", sessions=[], source_id="course_table", covers=covers, parent_id="course_ranking",
            outline_path="Course comparison", text=ranking, embed_text=ranking,
        )
        hits.append(Hit(chunk=chunk, rank=1, label="S1"))
    # One source per course, built from its row.
    for row in selected:
        components = _components(row.get("components", ""))
        course = CourseRow(
            course_code=row["course_code"],
            course_name=row["course_name"],
            final_exam_weight=_optional_float(row.get("final_exam_weight", "")),
            components=json.dumps(components, ensure_ascii=False),
            attendance_limit_pct=_optional_float(row.get("attendance_limit_pct", "")),
            retake_policy=row.get("retake_policy", ""),
            professor=row.get("professor", ""),
            n_sessions=_optional_int(row.get("n_sessions", "")),
            language=row.get("language", ""),
            evidence_section_ids=row.get("evidence_section_ids", ""),
            human_checked=row.get("human_checked", "").strip().lower() in {"true", "1", "yes"},
        )
        rank = len(hits) + 1
        covers = [value for value in course.evidence_section_ids.split(";") if value]
        text = _row_text(course)
        # The same chunk_id as the course summary chunk of the knowledge base, so the app shows both the same way.
        chunk = Chunk(
            chunk_id=f"course_row__{course.course_code}",
            kb_sig=cfg.kb_sig,
            strategy="section",
            doc_type="course_row",
            course_code=course.course_code,
            sessions=[],
            source_id=course.course_code,
            covers=covers,
            parent_id=f"course_row__{course.course_code}",
            outline_path=course.course_name,
            text=text,
            embed_text=text,
        )
        hits.append(Hit(chunk=chunk, rank=rank, label=f"S{rank}"))
    return hits


# The rows of the course table (course_code, course_name...), or an empty list when it is not built.
# answer.py uses them to find which courses a question names.
def course_rows() -> list[dict[str, str]]:
    path = PATHS["course_table"]
    return list(_load_rows(*_file_key(path))) if path.is_file() else []


# True when the question is answered by the ranking line: it compares a column the table holds, uses a
# ranking word and asks for nothing else. answer.py uses it to send such questions to the table.
def asks_for_ranking(question: str) -> bool:
    return bool(_compared_field(question) and _RANKING.search(question) and not _OTHER_ASK.search(question))


# Empty the cached rows, so the table is read again after it is rebuilt.
def clear_table_cache() -> None:
    _load_rows.cache_clear()
