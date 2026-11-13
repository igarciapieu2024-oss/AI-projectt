"""Cross-course questions answered from the course table.

Owner: Sergio.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from src.config import PATHS
from src.retrieval.filters import mentioned_course_codes
from src.schemas import Chunk, CourseRow, Hit
from src.utils.io import read_csv


def _file_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=2)
def _load_rows(path: str, modified_ns: int, size: int) -> tuple[dict[str, str], ...]:
    del modified_ns, size
    return tuple(read_csv(path))


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


def _optional_float(value: str) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"course table contains an invalid numeric value: {value!r}") from exc


def _optional_int(value: str) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"course table contains an invalid session count: {value!r}") from exc


def _row_text(row: CourseRow) -> str:
    parts = [f"{row.course_name} ({row.course_code})"]
    if row.final_exam_weight is not None:
        parts.append(f"final exam {row.final_exam_weight:g}%")
    else:
        # said explicitly: without it the model read another component (e.g. "Intermediate Tests 60%") as the final exam
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


# which column a comparison question is about, and the words that ask for the top or the bottom of it
_SORT_FIELDS = [
    (re.compile(r"final exam", re.I), "final_exam_weight"),
    (re.compile(r"attendance|absence|absences|miss", re.I), "attendance_limit_pct"),
    (re.compile(r"sessions|classes|long", re.I), "n_sessions"),
]
_ASCENDING = re.compile(r"\b(lowest|least|smallest|fewest|minimum|less|shortest)\b", re.I)


# how each compared column is named and printed in the ranking line
_FIELD_WORDS = {
    "final_exam_weight": ("final exam weight", "%", "no final exam"),
    "attendance_limit_pct": ("minimum attendance", "%", "not stated in the syllabus"),
    "n_sessions": ("number of sessions", "", "not stated in the syllabus"),
}


def _compared_field(question: str) -> str | None:
    return next((name for pattern, name in _SORT_FIELDS if pattern.search(question)), None)


def _ranking_text(question: str, rows: list[dict[str, str]]) -> str | None:
    """The answer to a comparison question written out, e.g. "Highest final exam weight: Strategies (45%). ...".

    Sorting the rows was not enough: the small local model skipped S1 ("Management Control and Organizational
    Behaviour, both 40%") or merged rows ("Strategies and Organizational Behaviour, both 45%"; OB is 40%). So the
    first sentence states the top course(s) on its own, ties grouped, and the rest follows in order.
    """
    field = _compared_field(question)
    if field is None or len(rows) < 2:
        return None
    label, unit, missing = _FIELD_WORDS[field]
    lowest = bool(_ASCENDING.search(question))
    groups: list[tuple[float, list[str]]] = []  # rows arrive sorted: equal values become one group
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

    def names(group):
        return " and ".join(group[1]) if len(group[1]) == 2 else ", ".join(group[1])

    top = groups[0]
    text = f"{'Lowest' if lowest else 'Highest'} {label}: {names(top)} ({top[0]:g}{unit})."
    if len(groups) > 1:
        text += " Then: " + "; ".join(f"{names(g)} ({g[0]:g}{unit})" for g in groups[1:]) + "."
    if without:
        text += f" {', '.join(without)}: {missing}."
    return text


def _ordered_for(question: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Sort the course rows by the column the question compares, so S1 is the answer and the model only reads it.

    The small local model mixed up the numbers when it had to compare 7 rows itself
    ("Strategies and Management Control, both 40%"; Strategies is 45%). Rows without a value go last.
    """
    field = _compared_field(question)
    if field is None:
        return rows
    ascending = bool(_ASCENDING.search(question))

    def key(row):
        try:
            value = float(row.get(field) or "")
        except ValueError:
            return (1, 0.0)  # no value: always at the end
        return (0, value if ascending else -value)

    return sorted(rows, key=key)


def answer_from_table(question: str, cfg) -> list[Hit]:
    """Return table-row evidence for every course, or courses named in the question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be empty")
    path = PATHS["course_table"]
    if not path.is_file():
        return []
    rows = list(_load_rows(*_file_key(path)))
    named_codes = set(mentioned_course_codes(question, rows))
    selected = [row for row in rows if not named_codes or row["course_code"] in named_codes]
    selected = _ordered_for(question, selected)
    hits = []
    ranking = _ranking_text(question, selected)
    if ranking:
        # S1 = the ranking already written out; the course rows follow as S2, S3... for the details and citations
        covers = sorted({value for row in selected for value in row.get("evidence_section_ids", "").split(";") if value})
        chunk = Chunk(
            chunk_id="course_ranking", kb_sig=cfg.kb_sig, strategy="section", doc_type="course_row",
            course_code="", sessions=[], source_id="course_table", covers=covers, parent_id="course_ranking",
            outline_path="Course comparison", text=ranking, embed_text=ranking,
        )
        hits.append(Hit(chunk=chunk, rank=1, label="S1"))
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


def clear_table_cache() -> None:
    """Clear cached course rows after rebuilding the course table."""
    _load_rows.cache_clear()
