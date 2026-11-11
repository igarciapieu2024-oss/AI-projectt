"""Parse syllabus PDFs into the shared SyllabusSection JSONL contract.

Output: data/interim/syllabus_sections.jsonl
One record is written per detected syllabus section. Tables are stored as Markdown.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pymupdf as fitz  # PyMuPDF (modern import; fitz alias retained below)

from src.config import PATHS
from src.schemas import SyllabusSection
from src.utils.ids import section_id
from src.utils.io import read_csv, write_jsonl


# Aliases are normalized before lookup, so punctuation, spaces, and capitalization
# do not affect classification. Values must match the shared schema enum.
_HEADING_ALIASES = {
    # Professor / instructor information
    "PROFESSOR": "PROFESSOR",
    "INSTRUCTOR": "PROFESSOR",
    "OFFICEHOURS": "PROFESSOR",
    "PROFESSOROFFICEHOURS": "PROFESSOR",

    # General course information (retained as OTHER by the shared schema)
    "SUBJECTDESCRIPTION": "OTHER",
    "COURSEDESCRIPTION": "OTHER",
    "COURSEOVERVIEW": "OTHER",
    "LEARNINGOBJECTIVES": "OTHER",
    "LEARNINGOUTCOMES": "OTHER",
    "COURSEOBJECTIVES": "OTHER",
    "COURSEOUTCOMES": "OTHER",

    # Dedicated categories
    "TEACHINGMETHODOLOGY": "METHODOLOGY",
    "LEARNINGMETHODOLOGY": "METHODOLOGY",
    "METHODOLOGY": "METHODOLOGY",
    "AIPOLICY": "AI_POLICY",
    "GENERATIVEAIPOLICY": "AI_POLICY",
    "ARTIFICIALINTELLIGENCEPOLICY": "AI_POLICY",
    "PROGRAM": "PROGRAM",
    "PROGRAMME": "PROGRAM",
    "COURSEPROGRAM": "PROGRAM",
    "COURSECONTENT": "PROGRAM",
    "EVALUATION": "EVALUATION",
    "EVALUATIONCRITERIA": "EVALUATION",
    "ASSESSMENT": "EVALUATION",
    "ASSESSMENTCRITERIA": "EVALUATION",
    "GRADING": "EVALUATION",
    "GRADINGCRITERIA": "EVALUATION",
    "RESITPOLICY": "RESIT",
    "RESITRETAKEPOLICY": "RESIT",
    "RESITANDRETAKEPOLICY": "RESIT",
    "RETAKEPOLICY": "RESIT",
    "BIBLIOGRAPHY": "BIBLIOGRAPHY",
    "REFERENCES": "BIBLIOGRAPHY",
    "READINGLIST": "BIBLIOGRAPHY",
    "ATTENDANCE": "ATTENDANCE",
    "ATTENDANCEPOLICY": "ATTENDANCE",

    # These are useful content sections but have no dedicated schema enum.
    "BEHAVIORRULES": "OTHER",
    "BEHAVIOURRULES": "OTHER",
    "ETHICALPOLICY": "OTHER",
    "ETHICSPOLICY": "OTHER",
    "ACADEMICINTEGRITY": "OTHER",
    "POLICIES": "OTHER",
}

_FOOTER_DATE = re.compile(
    r"^\d{1,2}(st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}$",
    re.IGNORECASE,
)


def _normalize_heading(text: str) -> str:
    """Normalize a heading so punctuation and spacing do not affect matching."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _heading_for_line(text: str) -> str | None:
    """Classify common syllabus headings without guessing from arbitrary prose."""
    stripped = text.strip()
    normalized = _normalize_heading(stripped)

    # Bibliography is especially prone to false positives because programme
    # descriptions often contain phrases such as "(See Bibliography)".
    # Accept only a clean standalone heading, not a parenthetical reference.
    if normalized == "BIBLIOGRAPHY":
        if stripped.endswith((")", "]", ".")) or stripped.startswith(("(", "[")):
            return None
        return "BIBLIOGRAPHY"

    # Exact aliases are safest and handle punctuation / spacing variations.
    if normalized in _HEADING_ALIASES:
        return _HEADING_ALIASES[normalized]

    # Common extracted variants with a prefix, suffix, or heading punctuation.
    # Keep these patterns anchored to avoid classifying ordinary body sentences.
    patterns = (
        (r"^(COURSE|SUBJECT)DESCRIPTION$", "OTHER"),
        (r"^(LEARNING|COURSE)OBJECTIVES?$", "OTHER"),
        (r"^(LEARNING|COURSE)OUTCOMES?$", "OTHER"),
        (r"^(TEACHING|LEARNING)?METHODOLOGY$", "METHODOLOGY"),
        (r"^(GENERATIVE|ARTIFICIALINTELLIGENCE|AI)POLICY$", "AI_POLICY"),
        (r"^(COURSE)?(PROGRAM|PROGRAMME|CONTENT)$", "PROGRAM"),
        (r"^(EVALUATION|ASSESSMENT|GRADING)(CRITERIA)?$", "EVALUATION"),
        (r"^(RE)?SIT(ANDRETAKE|RETAKE)?POLICY$", "RESIT"),
        (r"^(BIBLIOGRAPHY|REFERENCES|READINGLIST)$", "BIBLIOGRAPHY"),
        (r"^ATTENDANCE(POLICY)?$", "ATTENDANCE"),
        (r"^(PROFESSOR|INSTRUCTOR|OFFICEHOURS)$", "PROFESSOR"),
    )
    for pattern, label in patterns:
        if re.fullmatch(pattern, normalized):
            return label

    # "Professor: Name" is usually extracted as a single line.
    if re.match(r"^(professor|instructor)\s*:", stripped, re.IGNORECASE):
        return "PROFESSOR"

    return None


def _is_page_artifact(text: str, y0: float, page_height: float) -> bool:
    """Remove known recurring footer artifacts without deleting numbered lists."""
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.casefold() == "edited by documentation".casefold():
        return True
    # Dates can be meaningful in the body (deadlines, exam dates, etc.).
    # Treat a date as a footer artifact only when it is physically near
    # the bottom edge of the page.
    if _FOOTER_DATE.fullmatch(stripped) and y0 >= page_height - 42:
        return True

    # Page numbers in these syllabus PDFs sit close to the bottom edge.
    # Do not remove numeric list markers such as "1." or "2." from body text.
    if re.fullmatch(r"\d{1,3}", stripped) and y0 >= page_height - 42:
        return True

    return False


def _markdown_table(rows: list[list[Any]]) -> str:
    """Convert a PyMuPDF table extraction into a compact Markdown table."""
    cleaned = []
    for row in rows:
        cleaned.append([
            re.sub(r"\s+", " ", str(cell or "")).strip().replace("|", r"\|")
            for cell in row
        ])

    # Remove completely empty rows.
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return ""

    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]

    header = cleaned[0]
    separator = ["---"] * width
    body = cleaned[1:]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)



def _merge_markdown_tables(tables: list[str]) -> list[str]:
    """Join page-split fragments when their Markdown headers are identical."""
    merged: list[str] = []
    for table in tables:
        lines = [line for line in table.splitlines() if line.strip()]
        if len(lines) < 2:
            merged.append(table)
            continue

        if merged:
            previous_lines = [line for line in merged[-1].splitlines() if line.strip()]
            if (
                len(previous_lines) >= 2
                and previous_lines[0] == lines[0]
                and previous_lines[1] == lines[1]
            ):
                # Same table continued on the next page: append data rows only.
                merged[-1] = "\n".join(previous_lines + lines[2:])
                continue

        merged.append(table)
    return merged


def _merge_adjacent_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive sections with the same schema category.

    This combines professor + office-hours records and evaluation continuations,
    while preserving distinct categories and page ranges.
    """
    merged: list[dict[str, Any]] = []
    for section in sections:
        # OTHER is a catch-all schema label. Keep consecutive OTHER headings
        # separate so distinct syllabus topics (e.g. course description and
        # learning objectives) do not collapse into one record.
        if (
            merged
            and merged[-1]["section_name"] == section["section_name"]
            and section["section_name"] != "OTHER"
        ):
            previous = merged[-1]
            previous["page_end"] = max(previous["page_end"], section["page_end"])
            previous["text_lines"].extend(section["text_lines"])
            previous["tables"].extend(section["tables"])
        else:
            merged.append({
                "section_name": section["section_name"],
                "page_start": section["page_start"],
                "page_end": section["page_end"],
                "text_lines": list(section["text_lines"]),
                "tables": list(section["tables"]),
            })

    for section in merged:
        section["tables"] = _merge_markdown_tables(section["tables"])

        # PyMuPDF can return a page-2 methodology table with its header and a
        # page-3 continuation containing only remaining rows. Join only when
        # the preceding table is the known three-column Learning Activity table.
        if section["section_name"] == "METHODOLOGY" and len(section["tables"]) > 1:
            combined = [section["tables"][0]]
            for fragment in section["tables"][1:]:
                if _is_methodology_continuation(fragment):
                    combined[0] = _append_table_rows(combined[0], fragment)
                else:
                    combined.append(fragment)
            section["tables"] = combined
    return merged


def _is_evaluation_matrix(table: str) -> bool:
    """Identify the evaluation matrix from its header, not from nearby prose."""
    first_line = next((line for line in table.splitlines() if line.startswith("|")), "")
    normalized = _normalize_heading(first_line)
    return all(token in normalized for token in ("CRITERIA", "PERCENTAGE", "LEARNINGOBJECTIVES", "COMMENTS"))


def _is_evaluation_continuation(table: str) -> bool:
    """Recognize a headerless continuation of the four-column evaluation matrix."""
    rows = [line for line in table.splitlines() if line.startswith("|")]
    if not rows:
        return False
    cells = [cell.strip() for cell in rows[0].strip("|").split("|")]
    if len(cells) != 4:
        return False
    first = _normalize_heading(cells[0])
    return first in {
        "FINALEXAM", "CLASSPARTICIPATION", "INTERMEDIATETESTS",
        "GROUPPRESENTATION", "INDIVIDUALWORK", "MIDTERMEXAM",
    }


def _is_methodology_continuation(table: str) -> bool:
    """Identify a headerless continuation of the methodology workload table."""
    lines = [line for line in table.splitlines() if line.startswith("|")]
    if len(lines) < 2:
        return False
    first_cells = [cell.strip() for cell in lines[0].strip("|").split("|")]
    # A continuation starts with activity names (e.g. Individual studying / TOTAL),
    # rather than the known three-column header.
    first = _normalize_heading(first_cells[0]) if first_cells else ""
    return first in {"INDIVIDUALSTUDYING", "TOTAL"}


def _append_table_rows(existing: str, continuation: str) -> str:
    """Append data rows from a headerless table fragment to an existing table."""
    old_lines = [line for line in existing.splitlines() if line.strip()]
    new_lines = [line for line in continuation.splitlines() if line.strip()]
    if len(old_lines) >= 2 and len(new_lines) >= 1:
        return "\n".join(old_lines + new_lines)
    return existing


def _page_events(page: fitz.Page) -> list[tuple[float, int, str, Any]]:
    """Return text-line and table events in vertical page order.

    Text inside detected table bounding boxes is omitted from the prose stream;
    the table is preserved separately as Markdown. This prevents table headers
    such as "Learning Objectives" from being mistaken for section headings.
    """
    events: list[tuple[float, int, str, Any]] = []
    table_events: list[tuple[float, int, str, Any]] = []
    table_rects = []

    # Detect tables first so their cell text can be excluded from the prose.
    try:
        found_tables = page.find_tables()
        for table in found_tables.tables:
            rows = table.extract()
            markdown = _markdown_table(rows or [])
            if markdown:
                bbox = fitz.Rect(table.bbox)
                table_rects.append(bbox)
                table_events.append((float(bbox.y0), 1, "table", markdown))
    except (AttributeError, RuntimeError, ValueError):
        # Keep text extraction usable if a particular page contains a table
        # that PyMuPDF cannot reliably detect.
        pass

    page_dict = page.get_text("dict")
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            line_bbox = fitz.Rect(line.get("bbox", (0, 0, 0, 0)))
            # Table cells are represented by the Markdown table event instead.
            if any(line_bbox.intersects(table_rect) for table_rect in table_rects):
                continue

            text = "".join(span.get("text", "") for span in line.get("spans", []))
            text = re.sub(r"[ \t]+", " ", text).strip()
            y0 = float(line_bbox.y0)
            if not _is_page_artifact(text, y0, float(page.rect.height)):
                events.append((y0, 0, "text", text))

    events.extend(table_events)
    events.sort(key=lambda event: (event[0], event[1]))
    return events


def _parse_syllabus_pdf(
    pdf_path: str | Path,
    source_id: str,
    course_code: str,
) -> list[SyllabusSection]:
    """Parse one syllabus PDF. Kept separate so one file can be tested first."""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Syllabus PDF not found: {pdf_path}")

    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def start_section(section_name: str, page_number: int) -> None:
        nonlocal current
        current = {
            "section_name": section_name,
            "page_start": page_number,
            "page_end": page_number,
            "text_lines": [],
            "tables": [],
        }
        sections.append(current)

    def add_text(text: str, page_number: int) -> None:
        nonlocal current
        if current is None:
            # Preserve front-matter text (course title and basic course details).
            start_section("OTHER", page_number)
        current["text_lines"].append(text)
        current["page_end"] = page_number

    def add_table(markdown: str, page_number: int) -> None:
        nonlocal current

        # The Management Control evaluation matrix begins with its header at the
        # bottom of one page and continues on the next. Associate it with the
        # nearest preceding EVALUATION section, even if intervening extracted
        # table text caused a temporary OTHER record.
        target = None
        if _is_evaluation_matrix(markdown) or _is_evaluation_continuation(markdown):
            target = next(
                (section for section in reversed(sections)
                 if section["section_name"] == "EVALUATION"),
                None,
            )

        if target is None and current is None:
            start_section("OTHER", page_number)
            target = current
        elif target is None:
            target = current

        target["tables"].append(markdown)
        target["page_end"] = max(target["page_end"], page_number)

    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            for _, _, event_type, payload in _page_events(page):
                if event_type == "text":
                    heading = _heading_for_line(payload)
                    if heading is not None:
                        start_section(heading, page_number)
                    add_text(payload, page_number)
                elif event_type == "table":
                    add_table(payload, page_number)

    # Collapse adjacent same-category fragments only after all pages have been
    # read. This keeps section boundaries stable while joining office hours and
    # multi-page evaluation/methodology continuations.
    sections = _merge_adjacent_sections(sections)

    records: list[SyllabusSection] = []
    for index, section in enumerate(sections, start=1):
        text = "\n".join(section["text_lines"]).strip()
        tables = section["tables"]

        # Do not emit empty records.
        if not text and not tables:
            continue

        records.append(
            SyllabusSection(
                section_id=section_id(source_id, index),
                source_id=source_id,
                course_code=course_code,
                section_name=section["section_name"],
                page_start=section["page_start"],
                page_end=section["page_end"],
                text=text,
                tables=tables,
            )
        )

    return records


def parse_syllabi(cfg) -> int:
    """Parse every syllabus in the manifest and write SyllabusSection JSONL rows."""
    del cfg  # The shared PATHS object is rooted at the repository, as in config.py.

    manifest_path = Path(PATHS["manifest"])
    raw_root = Path(PATHS["root"]) / "data" / "raw"
    output_path = Path(PATHS["syllabus_sections"])

    manifest_rows = read_csv(manifest_path)
    syllabus_rows = [
        row for row in manifest_rows
        if row.get("doc_type", "").strip().lower() == "syllabus"
    ]

    if not syllabus_rows:
        raise ValueError(f"No syllabus rows found in manifest: {manifest_path}")

    all_sections: list[SyllabusSection] = []
    for row in syllabus_rows:
        relative_file = row.get("file", "").strip()
        if not relative_file:
            raise ValueError(f"Empty file path for source_id={row.get('source_id')}")

        pdf_path = raw_root / Path(relative_file)
        all_sections.extend(
            _parse_syllabus_pdf(
                pdf_path=pdf_path,
                source_id=row["source_id"],
                course_code=row["course_code"],
            )
        )

    return write_jsonl(output_path, all_sections)
