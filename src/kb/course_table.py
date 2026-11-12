"""One CourseRow per course from the syllabus sections -> data/processed/course_table.csv.

Owner: Santiago.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict

from src.config import PATHS
from src.schemas import CourseRow, SyllabusSection
from src.utils.io import read_csv, read_jsonl, write_csv

logger = logging.getLogger(__name__)

_WEIGHT = re.compile(
    r"(?<![\w.])(?P<weight>\d{1,3}(?:[.,]\d+)?)\s*(?:%|percent(?:age)?)(?![A-Za-z])",
    re.IGNORECASE,
)
_COMPONENT_ALIASES = {
    "finalexam": "Final Exam",
    "finalexamination": "Final Exam",
    "intermediatetest": "Intermediate Test",
    "intermediatetests": "Intermediate Tests",
    "individualwork": "Individual Work",
    "midtermexam": "Midterm",
    "midtermexams": "Midterms",
    "midterm": "Midterm",
    "classtest": "Class Test",
    "classtests": "Class Tests",
    "classparticipation": "Class Participation",
}
_COMPONENT_TERMS = re.compile(
    r"\b(exams?|examinations?|tests?|quizzes|assignments?|participation|projects?|"
    r"presentations?|coursework|assessments?|midterms?|finals?|practicals?|"
    r"exercises?|cases?|papers?|reports?|work)\b",
    re.IGNORECASE,
)

_NON_COMPONENT_PHRASES = re.compile(
    r"\b(?:personal|work reasons|will have|weight of|missing|absence|absent|"
    r"grade of|in case|because)\b",
    re.IGNORECASE,
)


def _weight_value(match: re.Match[str]) -> float:
    return float(match.group("weight").replace(",", "."))


def _clean_component_name(value: str) -> str:
    value = re.sub(r"^\s*(?:[A-Z]|\d+)[.)]\s*", "", value.strip())
    value = re.sub(r"\s*\([^)]*$", "", value)
    value = re.sub(r"[\s:–—-]+$", "", value).strip(" .")
    value = re.sub(r"^(?:the\s+)?", "", value, flags=re.IGNORECASE)
    value = re.sub(
        r"\s+(?:is|are|worth|represents?|accounts?\s+for)\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    compact = re.sub(r"[^a-z]", "", value.casefold())
    if compact in _COMPONENT_ALIASES:
        return _COMPONENT_ALIASES[compact]
    return " ".join(word.capitalize() for word in value.split())


def _component(name: str, weight: float) -> dict[str, float | str] | None:
    name = _clean_component_name(name)
    if not name or not re.search(r"[A-Za-z]", name) or not 0 <= weight <= 100:
        return None
    return {"name": name, "weight": weight}


def _table_components(table: str) -> list[dict[str, float | str]]:
    components = []
    for line in table.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip().replace(r"\|", "|") for cell in line.strip().strip("|").split("|")]
        if not cells or all(re.fullmatch(r"[:\-\s]+", cell or "-") for cell in cells):
            continue
        matches = [(index, match) for index, cell in enumerate(cells) for match in _WEIGHT.finditer(cell)]
        if not matches:
            continue
        weight_index, weight_match = matches[0]
        name = cells[weight_index][:weight_match.start()]
        if not name.strip():
            name = next(
                (cell for cell in reversed(cells[:weight_index]) if cell.strip()),
                "",
            )
        item = _component(name, _weight_value(weight_match))
        if item:
            components.append(item)
    return components


def _text_components(text: str, *, require_component_term: bool) -> list[dict[str, float | str]]:
    components = []
    segments = re.split(r"\n|;|…|\.{2,}|•", text)
    previous_label = ""
    for segment in segments:
        match = _WEIGHT.search(segment)
        if not match:
            previous_label = segment.strip()
            continue
        name = segment[:match.start()]
        if not name.strip() and previous_label:
            name = previous_label
        previous_label = ""
        if require_component_term and not _COMPONENT_TERMS.search(name):
            continue
        if _NON_COMPONENT_PHRASES.search(name) or len(name.split()) > 6:
            continue
        item = _component(name, _weight_value(match))
        if item:
            components.append(item)
    return components


def _component_key(component: dict[str, float | str]) -> str:
    key = re.sub(r"[^a-z]", "", str(component["name"]).casefold())
    return "participation" if key in {"participation", "classparticipation"} else key


def _contains_component(
    components: list[dict[str, float | str]],
    candidate: dict[str, float | str],
) -> bool:
    return any(
        _component_key(item) == _component_key(candidate)
        and abs(float(item["weight"]) - float(candidate["weight"])) < 0.01
        for item in components
    )


def _attendance_value(sections: list[SyllabusSection]) -> tuple[float | None, str | None]:
    attendance_pattern = re.compile(
        r"\battendance\b[^\n.;]{0,60}?(?P<after>\d{1,3}(?:[.,]\d+)?)\s*%"
        r"|(?P<before>\d{1,3}(?:[.,]\d+)?)\s*%[^\n.;]{0,40}\battendance\b",
        re.IGNORECASE,
    )
    for section in sections:
        if section.section_name not in {"ATTENDANCE", "RESIT"}:
            continue
        content = "\n".join([section.text, *section.tables])
        match = attendance_pattern.search(content)
        if match:
            value = match.group("after") or match.group("before")
            return float(value.replace(",", ".")), section.section_id
    return None, None


def _retake_policy(sections: list[SyllabusSection]) -> tuple[str, str | None]:
    title_pattern = re.compile(
        r"^\s*(?:re[\s-]?sit|re[\s-]?take|retake)"
        r"(?:\s*/\s*(?:re[\s-]?sit|re[\s-]?take|retake))?\s+policy\s*$",
        re.IGNORECASE,
    )
    for section in sections:
        if section.section_name != "RESIT":
            continue
        lines = [line.strip(" |") for line in section.text.splitlines()]
        lines = [line for line in lines if line and not title_pattern.fullmatch(line)]
        content = re.sub(r"\s+", " ", " ".join(lines)).strip()
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", content)
            if sentence.strip()
        ]
        selected = []
        for sentence in sentences[:3]:
            if selected and sum(len(value) + 1 for value in selected) + len(sentence) > 300:
                break
            selected.append(sentence)
        if selected:
            return " ".join(selected), section.section_id
    return "", None


def _professor(sections: list[SyllabusSection]) -> tuple[str, str | None]:
    professor_pattern = re.compile(
        r"^\s*(?:professor|instructor)\s*:\s*(.+?)\s*$",
        re.IGNORECASE,
    )
    email_pattern = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
    for section in sections:
        if section.section_name != "PROFESSOR":
            continue
        for line in section.text.splitlines():
            match = professor_pattern.match(line)
            if match:
                name = email_pattern.sub("", match.group(1)).strip(" \t,;()")
                if name:
                    return name, section.section_id
    return "", None


def _session_count(sections: list[SyllabusSection]) -> tuple[int | None, str | None]:
    session_pattern = re.compile(
        r"\bSESSIONS?\s+(\d+)(?:\s*(?:-|–|—|to)\s*(\d+))?\b",
        re.IGNORECASE,
    )
    highest: int | None = None
    source_id: str | None = None
    for section in sections:
        if section.section_name != "PROGRAM":
            continue
        content = "\n".join([section.text, *section.tables])
        for match in session_pattern.finditer(content):
            values = [int(value) for value in match.groups() if value is not None]
            candidate = max(values)
            if highest is None or candidate > highest:
                highest = candidate
                source_id = section.section_id
    return highest, source_id


def _course_row(
    course_code: str,
    manifest_row: dict[str, str],
    sections: list[SyllabusSection],
) -> CourseRow:
    evaluation_sections = [section for section in sections if section.section_name == "EVALUATION"]
    table_components: list[dict[str, float | str]] = []
    table_sources: list[str] = []
    for section in evaluation_sections:
        for table in section.tables:
            for item in _table_components(table):
                if not _contains_component(table_components, item):
                    table_components.append(item)
                    table_sources.append(section.section_id)

    table_total = sum(float(item["weight"]) for item in table_components)
    text_components = []
    text_sources = []
    for section in evaluation_sections:
        for item in _text_components(section.text, require_component_term=True):
            if not _contains_component(text_components, item):
                text_components.append(item)
                text_sources.append(section.section_id)
    text_total = sum(float(item["weight"]) for item in text_components)

    text_is_complete = bool(text_components) and abs(text_total - 100) <= 1
    text_and_table_overlap = any(
        _contains_component(table_components, item)
        for item in text_components
    )
    if text_is_complete and (
        not table_components or abs(table_total - 100) > 1 or text_and_table_overlap
    ):
        components = text_components
        component_sources = text_sources
    elif table_components and abs(table_total - 100) <= 1:
        components = table_components
        component_sources = table_sources
    else:
        # Some extracted tables spill their final label and percentage into OTHER.
        components = table_components
        component_sources = table_sources
        for section in sections:
            if section.section_name != "OTHER":
                continue
            continuation = _text_components(section.text, require_component_term=True)
            combined = list(components)
            added = []
            for item in continuation:
                if not _contains_component(combined, item):
                    combined.append(item)
                    added.append(item)
            combined_total = sum(float(item["weight"]) for item in combined)
            if added and abs(combined_total - 100) <= 1:
                components = combined
                component_sources.extend([section.section_id] * len(added))
                break
        if not components:
            components = text_components
            component_sources = text_sources

    final_exam_weight = next(
        (
            float(item["weight"])
            for item in components
            if re.search(r"\bfinal\s+(?:exam|examination)\b", str(item["name"]), re.IGNORECASE)
        ),
        None,
    )

    attendance_limit_pct, attendance_source = _attendance_value(sections)
    retake_policy, retake_source = _retake_policy(sections)
    professor, professor_source = _professor(sections)
    n_sessions, sessions_source = _session_count(sections)

    evidence_ids = set(component_sources)
    for source_id in (attendance_source, retake_source, professor_source, sessions_source):
        if source_id:
            evidence_ids.add(source_id)
    evidence_section_ids = ";".join(
        section.section_id for section in sections if section.section_id in evidence_ids
    )

    if components and abs(sum(float(item["weight"]) for item in components) - 100) > 1:
        logger.warning(
            "%s: evaluation components sum to %.1f%%, not approximately 100%%",
            course_code,
            sum(float(item["weight"]) for item in components),
        )

    return CourseRow(
        course_code=course_code,
        course_name=manifest_row.get("course_name", "").strip(),
        final_exam_weight=final_exam_weight,
        components=json.dumps(components, ensure_ascii=False),
        attendance_limit_pct=attendance_limit_pct,
        retake_policy=retake_policy,
        professor=professor,
        n_sessions=n_sessions,
        language=manifest_row.get("language", "").strip(),
        evidence_section_ids=evidence_section_ids,
    )


def build_course_table(cfg) -> int:
    """Extract and validate one row per course; return the row count."""
    del cfg
    manifest_rows = read_csv(PATHS["manifest"])
    courses: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        course_code = row.get("course_code", "").strip()
        if not course_code:
            raise ValueError("manifest.csv contains a row without course_code")
        if course_code not in courses or row.get("doc_type", "").strip().lower() == "syllabus":
            courses[course_code] = row

    if not courses:
        raise ValueError(f"No courses found in manifest: {PATHS['manifest']}")

    section_records = read_jsonl(PATHS["syllabus_sections"], SyllabusSection)
    sections_by_course: dict[str, list[SyllabusSection]] = defaultdict(list)
    for section in section_records:
        sections_by_course[section.course_code].append(section)

    rows = [
        _course_row(course_code, manifest_row, sections_by_course[course_code])
        for course_code, manifest_row in courses.items()
    ]
    return write_csv(PATHS["course_table"], rows, CourseRow.field_names())
