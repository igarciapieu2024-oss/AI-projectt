# Read the syllabus PDFs listed in the manifest and turn them into SyllabusSection objects. The output is
# data/interim/syllabus_sections.jsonl, with one record per syllabus section (EVALUATION, PROGRAM, BIBLIOGRAPHY...). A
# syllabus is organised by topic, not by page, so sections are a better unit than pages. We keep the tables
# separately, as Markdown. The course table (src/kb/course_table.py) and the chunking step (src/kb/chunking.py) read
# these records.

from __future__ import annotations

# Regular expressions for the section titles and the footers; Any is used in the types.
import re
from pathlib import Path
from typing import Any

import pymupdf as fitz  # PyMuPDF, imported under its older name "fitz"

# The folders of the project, the format of a syllabus section, the builder of section ids and the CSV reader and
# JSONL writer.
from src.config import PATHS
from src.schemas import SyllabusSection
from src.utils.ids import section_id
from src.utils.io import read_csv, write_jsonl


# The section titles found in the 7 syllabi, mapped to a fixed set of section labels. Each syllabus writes its titles
# differently, so we normalise them (upper case, no spaces or punctuation) and look them up here. The labels have to
# match the values SyllabusSection allows.
_HEADING_ALIASES = {
    # Professor / instructor information
    "PROFESSOR": "PROFESSOR",
    "INSTRUCTOR": "PROFESSOR",
    "OFFICEHOURS": "PROFESSOR",
    "PROFESSOROFFICEHOURS": "PROFESSOR",

    # General course information, kept under the catch-all label OTHER
    "SUBJECTDESCRIPTION": "OTHER",
    "COURSEDESCRIPTION": "OTHER",
    "COURSEOVERVIEW": "OTHER",
    "LEARNINGOBJECTIVES": "OTHER",
    "LEARNINGOUTCOMES": "OTHER",
    "COURSEOBJECTIVES": "OTHER",
    "COURSEOUTCOMES": "OTHER",

    # Sections with their own label
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

    # Useful sections that have no label of their own
    "BEHAVIORRULES": "OTHER",
    "BEHAVIOURRULES": "OTHER",
    "ETHICALPOLICY": "OTHER",
    "ETHICSPOLICY": "OTHER",
    "ACADEMICINTEGRITY": "OTHER",
    "POLICIES": "OTHER",
}

# A date, printed in the footer of some syllabus pages.
_FOOTER_DATE = re.compile(
    r"^\d{1,2}(st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}$",
    re.IGNORECASE,
)


# Put a line in upper case and remove everything that isn't a letter or a digit, so titles written in different ways
# can be compared.
def _normalize_heading(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


# Return the section label if the line is a known section title, otherwise None. We only accept short, exact matches,
# so normal sentences aren't taken as titles.
def _heading_for_line(text: str) -> str | None:
    # The line without spaces at the ends, and its normalised form.
    stripped = text.strip()
    normalized = _normalize_heading(stripped)

    # Bibliography gives many false positives, because programme descriptions often contain phrases such as "(See
    # Bibliography)". We only accept a clean standalone heading, not a reference in brackets.
    if normalized == "BIBLIOGRAPHY":
        if stripped.endswith((")", "]", ".")) or stripped.startswith(("(", "[")):
            return None
        return "BIBLIOGRAPHY"

    # Exact match with one of the known titles.
    if normalized in _HEADING_ALIASES:
        return _HEADING_ALIASES[normalized]

    # Variants of the same titles with an optional prefix or suffix ("COURSE PROGRAM", "ATTENDANCE POLICY"). The
    # patterns have to match the whole line, so ordinary sentences are never taken as titles.
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

    # "Professor: Name" is read as a single line.
    if re.match(r"^(professor|instructor)\s*:", stripped, re.IGNORECASE):
        return "PROFESSOR"

    return None


# True for lines that are page decoration and not content: empty lines, the "Edited by documentation" note, and footer
# dates or page numbers.
def _is_page_artifact(text: str, y0: float, page_height: float) -> bool:
    # An empty line is never content.
    stripped = text.strip()
    if not stripped:
        return True
    # A note some syllabi print on every page.
    if stripped.casefold() == "edited by documentation".casefold():
        return True
    # A date can be real content in the body, so we only treat it as a footer when it's within 42 points of the bottom
    # edge.
    if _FOOTER_DATE.fullmatch(stripped) and y0 >= page_height - 42:
        return True

    # Page numbers are close to the bottom edge too. Checking the position stops us from removing list numbers such as
    # "1." or "2." from the body text.
    if re.fullmatch(r"\d{1,3}", stripped) and y0 >= page_height - 42:
        return True

    return False


# Turn the rows of a table found by PyMuPDF into a Markdown table. Markdown keeps every value in its row and column,
# which plain text would lose.
def _markdown_table(rows: list[list[Any]]) -> str:
    # The rows with their cells cleaned.
    cleaned = []
    for row in rows:
        # Collapse the spaces inside each cell and escape "|", which is what separates the columns.
        cleaned.append([
            re.sub(r"\s+", " ", str(cell or "")).strip().replace("|", r"\|")
            for cell in row
        ])

    # Remove completely empty rows.
    cleaned = [row for row in cleaned if any(row)]
    # A table with no content gives no text.
    if not cleaned:
        return ""

    # Shorter rows are filled with empty cells, so every row has the same number of columns.
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]

    # The first row is the header, followed by the "| --- |" separator line.
    header = cleaned[0]
    separator = ["---"] * width
    body = cleaned[1:]

    # The header line, the separator line and then one line per row.
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)



# Join the tables that the PDF splits across two pages. Two tables in a row with the same header (their first two
# Markdown lines) count as one table.
def _merge_markdown_tables(tables: list[str]) -> list[str]:
    # The tables after joining, in order.
    merged: list[str] = []
    for table in tables:
        # The non-empty lines of this table; a table with fewer than two lines has no header to compare.
        lines = [line for line in table.splitlines() if line.strip()]
        if len(lines) < 2:
            merged.append(table)
            continue

        # Compare it with the previous table: the same first two lines mean it is the same table.
        if merged:
            previous_lines = [line for line in merged[-1].splitlines() if line.strip()]
            if (
                len(previous_lines) >= 2
                and previous_lines[0] == lines[0]
                and previous_lines[1] == lines[1]
            ):
                # The same table continued on the next page: we only add its data rows.
                merged[-1] = "\n".join(previous_lines + lines[2:])
                continue

        # A different table is added as it is.
        merged.append(table)
    return merged


# Join sections in a row that have the same label, for example "Professor" followed by "Office hours" (both
# PROFESSOR), or an evaluation section continued on the next page. We keep the first and the last page of the joined
# section.
def _merge_adjacent_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # The sections after joining, in order.
    merged: list[dict[str, Any]] = []
    for section in sections:
        # OTHER is a catch-all label. We keep OTHER sections in a row separate, so different topics (such as the
        # course description and the learning objectives) don't end up in one record.
        if (
            merged
            and merged[-1]["section_name"] == section["section_name"]
            and section["section_name"] != "OTHER"
        ):
            # The same label as the previous section: we extend the previous one with the pages, text and tables of
            # this one.
            previous = merged[-1]
            previous["page_end"] = max(previous["page_end"], section["page_end"])
            previous["text_lines"].extend(section["text_lines"])
            previous["tables"].extend(section["tables"])
        # A new section: a copy of it is added to the list.
        else:
            merged.append({
                "section_name": section["section_name"],
                "page_start": section["page_start"],
                "page_end": section["page_end"],
                "text_lines": list(section["text_lines"]),
                "tables": list(section["tables"]),
            })

    # Then, inside each section, the tables split across pages are joined too.
    for section in merged:
        section["tables"] = _merge_markdown_tables(section["tables"])

        # PyMuPDF can return a methodology table on page 2 with its header, and a continuation on page 3 with only the
        # remaining rows. We join them only when the fragment looks like the rest of that workload table.
        if section["section_name"] == "METHODOLOGY" and len(section["tables"]) > 1:
            combined = [section["tables"][0]]
            for fragment in section["tables"][1:]:
                if _is_methodology_continuation(fragment):
                    combined[0] = _append_table_rows(combined[0], fragment)
                else:
                    combined.append(fragment)
            section["tables"] = combined
    return merged


# True if the table is the evaluation matrix, which we recognise by its four column headers (criteria, percentage,
# learning objectives, comments).
def _is_evaluation_matrix(table: str) -> bool:
    # The first line of the table is its header.
    first_line = next((line for line in table.splitlines() if line.startswith("|")), "")
    normalized = _normalize_heading(first_line)
    return all(token in normalized for token in ("CRITERIA", "PERCENTAGE", "LEARNINGOBJECTIVES", "COMMENTS"))


# True if the table is the rest of the evaluation matrix on the next page: four columns and no header, starting
# directly with a grade component such as "Final exam".
def _is_evaluation_continuation(table: str) -> bool:
    # The lines of the table; without lines it isn't a continuation.
    rows = [line for line in table.splitlines() if line.startswith("|")]
    if not rows:
        return False
    # The cells of its first row: a continuation has four columns.
    cells = [cell.strip() for cell in rows[0].strip("|").split("|")]
    if len(cells) != 4:
        return False
    # And its first cell is the name of a grade component.
    first = _normalize_heading(cells[0])
    return first in {
        "FINALEXAM", "CLASSPARTICIPATION", "INTERMEDIATETESTS",
        "GROUPPRESENTATION", "INDIVIDUALWORK", "MIDTERMEXAM",
    }


# True if the table is the rest of the methodology workload table on the next page.
def _is_methodology_continuation(table: str) -> bool:
    # The lines of the table; a continuation has at least two.
    lines = [line for line in table.splitlines() if line.startswith("|")]
    if len(lines) < 2:
        return False
    first_cells = [cell.strip() for cell in lines[0].strip("|").split("|")]
    # A continuation starts with activity names instead of the table header.
    first = _normalize_heading(first_cells[0]) if first_cells else ""
    return first in {"INDIVIDUALSTUDYING", "TOTAL"}


# Add the rows of a table fragment that has no header to the end of an existing table.
def _append_table_rows(existing: str, continuation: str) -> str:
    # The non-empty lines of both tables.
    old_lines = [line for line in existing.splitlines() if line.strip()]
    new_lines = [line for line in continuation.splitlines() if line.strip()]
    # Join them only when the existing table has its header and the fragment has at least one row.
    if len(old_lines) >= 2 and len(new_lines) >= 1:
        return "\n".join(old_lines + new_lines)
    return existing


# Return the text lines and the tables of a page as "events", sorted from top to bottom, so each table stays in the
# section where it appears. Each event is (vertical position, kind order, "text" or "table", content). We skip the
# text inside a table, because the table is stored separately as Markdown; this also stops table headers such as
# "Learning Objectives" from being read as section titles.
def _page_events(page: fitz.Page) -> list[tuple[float, int, str, Any]]:
    # The text events, the table events and the areas of the page that tables occupy.
    events: list[tuple[float, int, str, Any]] = []
    table_events: list[tuple[float, int, str, Any]] = []
    table_rects = []

    # We find the tables first, so their cell text can be left out of the text lines.
    try:
        found_tables = page.find_tables()
        for table in found_tables.tables:
            rows = table.extract()
            markdown = _markdown_table(rows or [])
            if markdown:
                bbox = fitz.Rect(table.bbox)  # area of the page the table occupies
                table_rects.append(bbox)
                table_events.append((float(bbox.y0), 1, "table", markdown))
    except (AttributeError, RuntimeError, ValueError):
        # If PyMuPDF can't find the tables of a page, we still read its text.
        pass

    page_dict = page.get_text("dict")  # text blocks -> lines -> spans
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:  # skip image blocks
            continue
        for line in block.get("lines", []):
            # The area of the page this line occupies.
            line_bbox = fitz.Rect(line.get("bbox", (0, 0, 0, 0)))
            # Lines inside a table are already part of the Markdown table.
            if any(line_bbox.intersects(table_rect) for table_rect in table_rects):
                continue

            # The text of the line, with single spaces, and its vertical position.
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            text = re.sub(r"[ \t]+", " ", text).strip()
            y0 = float(line_bbox.y0)
            # Page decoration (footers, page numbers) is left out.
            if not _is_page_artifact(text, y0, float(page.rect.height)):
                events.append((y0, 0, "text", text))

    # Sort by vertical position; at the same height, text (0) comes before a table (1).
    events.extend(table_events)
    events.sort(key=lambda event: (event[0], event[1]))
    return events


# Read one syllabus and return one SyllabusSection per section. We go through the page events in order: a section
# title starts a new section, and text lines and tables are added to the current one.
def _parse_syllabus_pdf(
    pdf_path: str | Path,
    source_id: str,
    course_code: str,
) -> list[SyllabusSection]:
    # The path of the PDF.
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():  # stop early with a clear message if the PDF is missing
        raise FileNotFoundError(f"Syllabus PDF not found: {pdf_path}")

    # The sections found so far, in order.
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None  # the section being filled

    # Open a new, empty section with the given label.
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

    # Add a text line to the current section.
    def add_text(text: str, page_number: int) -> None:
        nonlocal current
        # nonlocal lets these inner functions change current, which belongs to _parse_syllabus_pdf().
        if current is None:
            # Text before the first title (the course name and its basic details) goes into an OTHER section.
            start_section("OTHER", page_number)
        current["text_lines"].append(text)
        current["page_end"] = page_number

    # Add a table to the section it belongs to.
    def add_table(markdown: str, page_number: int) -> None:
        nonlocal current

        # The Management Control evaluation matrix starts with its header at the bottom of one page and continues on
        # the next. Both parts go to the nearest EVALUATION section before them, even if some text read in between
        # opened a temporary OTHER section.
        target = None
        if _is_evaluation_matrix(markdown) or _is_evaluation_continuation(markdown):
            target = next(
                (section for section in reversed(sections)
                 if section["section_name"] == "EVALUATION"),
                None,
            )

        # Any other table goes to the section being filled; if there is none yet, an OTHER section is opened for it.
        if target is None and current is None:
            start_section("OTHER", page_number)
            target = current
        elif target is None:
            target = current

        # Add the table and extend the section to this page.
        target["tables"].append(markdown)
        target["page_end"] = max(target["page_end"], page_number)

    # Open the PDF and go through its pages in order.
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            page_number = page_index + 1  # page numbers start at 1, as in the PDF
            for _, _, event_type, payload in _page_events(page):
                if event_type == "text":
                    # A section title opens a new section; the title line itself is kept as the first line of that
                    # section.
                    heading = _heading_for_line(payload)
                    if heading is not None:
                        start_section(heading, page_number)
                    add_text(payload, page_number)
                elif event_type == "table":
                    add_table(payload, page_number)

    # We join the sections split across pages only after reading the whole PDF, so the section limits found above
    # don't change while we read.
    sections = _merge_adjacent_sections(sections)

    # Turn the sections into records, numbered from 1.
    records: list[SyllabusSection] = []
    for index, section in enumerate(sections, start=1):
        # The text of the section and its tables.
        text = "\n".join(section["text_lines"]).strip()
        tables = section["tables"]

        # Sections with no text and no tables aren't saved.
        if not text and not tables:
            continue

        records.append(
            SyllabusSection(
                # The id only depends on the document and the section number, so it stays the same after every
                # rebuild. The evaluation questions use these ids to name their correct source.
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


# Read every syllabus in the manifest, write data/interim/syllabus_sections.jsonl and return the number of sections
# written. src/pipeline.py calls it.
def parse_syllabi(cfg) -> int:
    del cfg  # not needed: all paths come from src.config.PATHS

    # The manifest, the folder with the PDFs and the file we write.
    manifest_path = Path(PATHS["manifest"])
    raw_root = Path(PATHS["root"]) / "data" / "raw"  # folder with the PDFs
    output_path = Path(PATHS["syllabus_sections"])

    # Every row of the manifest.
    manifest_rows = read_csv(manifest_path)
    # We keep only the syllabi; the slide decks are read by parse_slides.py.
    syllabus_rows = [
        row for row in manifest_rows
        if row.get("doc_type", "").strip().lower() == "syllabus"
    ]

    # A manifest without syllabi is a mistake.
    if not syllabus_rows:
        raise ValueError(f"No syllabus rows found in manifest: {manifest_path}")

    # The sections of all the syllabi together.
    all_sections: list[SyllabusSection] = []
    for row in syllabus_rows:
        relative_file = row.get("file", "").strip()
        if not relative_file:  # every manifest row must point to a PDF
            raise ValueError(f"Empty file path for source_id={row.get('source_id')}")

        # The path of the PDF, written in the manifest relative to data/raw.
        pdf_path = raw_root / Path(relative_file)
        all_sections.extend(
            _parse_syllabus_pdf(
                pdf_path=pdf_path,
                source_id=row["source_id"],
                course_code=row["course_code"],
            )
        )

    # Write all the sections and return how many there are.
    return write_jsonl(output_path, all_sections)
