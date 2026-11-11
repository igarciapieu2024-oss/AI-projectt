"""Parse slide PDFs into the shared SlideRecord JSONL contract.

Output: data/interim/slides.jsonl
One record is written per PDF page.
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from src.config import PATHS
from src.schemas import SlideRecord
from src.utils.ids import slide_id
from src.utils.io import read_csv, write_jsonl


# ---------------------------------------------------------------------------
# Text processing helpers


def _clean_line(text: str) -> str:
    """Normalize whitespace without removing meaningful content."""
    return re.sub(r"\s+", " ", text).strip()


def _extract_lines(page: pymupdf.Page) -> list[dict]:
    """Extract text lines with their position and font information."""
    result = []
    page_dict = page.get_text("dict")

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = _clean_line(
                "".join(span.get("text", "") for span in spans)
            )

            if not text:
                continue

            font_sizes = [
                float(span.get("size", 0))
                for span in spans
                if span.get("text", "").strip()
            ]

            result.append({
                "text": text,
                "x": float(line["bbox"][0]),
                "y": float(line["bbox"][1]),
                "font_size": max(font_sizes, default=0.0),
                "bold": any(
                    "bold" in span.get("font", "").lower()
                    for span in spans
                ),
            })

    return sorted(result, key=lambda item: (item["y"], item["x"]))


def _extract_text(lines: list[dict]) -> str:
    """Preserve all extracted text in reading order."""
    return "\n".join(item["text"] for item in lines).strip()


def _find_title(lines: list[dict], page_height: float) -> str:
    """Find a likely title among the lines near the top of the slide."""
    candidates = [
        item for item in lines
        if item["y"] <= page_height * 0.40
        and 2 <= len(item["text"]) <= 180
        and not re.fullmatch(r"\d+", item["text"])
    ]

    if not candidates:
        return ""

    # Prefer larger and bold text, while retaining top-to-bottom order.
    candidates.sort(
        key=lambda item: (
            -item["font_size"],
            not item["bold"],
            item["y"],
            item["x"],
        )
    )

    return candidates[0]["text"]


def _find_outline(lines: list[dict], title: str) -> list[str]:
    """Identify likely section headings using numbering and typography."""
    if not lines:
        return []

    sizes = sorted(
        item["font_size"] for item in lines if item["font_size"] > 0
    )
    median_size = sizes[len(sizes) // 2] if sizes else 0

    outline = []

    for item in lines:
        text = item["text"]

        if text == title or len(text) > 150:
            continue

        numbered = bool(
            re.match(
                r"^(?:\d+(?:\.\d+)*[.)]?|[A-Z][.)])\s+\S",
                text,
            )
        )

        typography_heading = (
            item["font_size"] >= median_size * 1.20
            and len(text) <= 110
        )

        if numbered or typography_heading:
            if text not in outline:
                outline.append(text)

    return outline


def _is_agenda(title: str, text: str) -> bool:
    """Detect common agenda and contents slides."""
    sample = f"{title}\n{text[:1500]}".lower()

    patterns = (
        r"\bagenda\b",
        r"\btable of contents\b",
        r"\bcontents\b",
        r"\boutline\b",
        r"\btoday'?s topics\b",
        r"\blearning objectives\b",
        r"\broadmap\b",
    )

    return any(re.search(pattern, sample) for pattern in patterns)


def _is_garbled(text: str) -> bool:
    """Flag pages with suspicious character extraction."""
    if not text:
        return False

    replacement_count = text.count("\ufffd")
    suspicious_count = sum(
        1 for char in text
        if ord(char) < 32 and char not in "\n\r\t"
    )

    total = max(len(text), 1)

    return (
        replacement_count / total > 0.01
        or suspicious_count / total > 0.01
    )


def _count_images(page: pymupdf.Page) -> int:
    """Count image occurrences on a page, without performing OCR."""
    try:
        return len(page.get_image_info(xrefs=True))
    except (AttributeError, RuntimeError):
        return len(page.get_images(full=True))



def _parse_sessions(value: str) -> list[int]:
    """Parse manifest sessions, e.g. '[1]', '[19,20]' or '5;6'."""
    if not value or not value.strip():
        return []

    # Remove optional square brackets and parse comma/semicolon separators.
    cleaned = value.strip().strip("[]")
    try:
        return [
            int(item.strip())
            for item in re.split(r"[,;]", cleaned)
            if item.strip()
        ]
    except ValueError as exc:
        raise ValueError(
            f"Invalid session value in manifest: {value!r}"
        ) from exc


# ---------------------------------------------------------------------------
# PDF parsing


def _parse_slide_pdf(
    pdf_path: str | Path,
    source_id: str,
    course_code: str,
    sessions: list[int],
) -> list[SlideRecord]:
    """Parse one slide deck, producing one record per page."""
    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"Slide PDF not found: {pdf_path}")

    records = []

    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            page_number = page_index + 1

            lines = _extract_lines(page)
            text = _extract_text(lines)

            title = _find_title(lines, float(page.rect.height))
            outline = _find_outline(lines, title)

            records.append(
                SlideRecord(
                    slide_id=slide_id(source_id, page_number),
                    source_id=source_id,
                    course_code=course_code,
                    sessions=sessions,
                    page=page_number,
                    title=title,
                    outline=outline,
                    text=text,
                    is_agenda=_is_agenda(title, text),
                    is_low_text=len(text.strip()) < 40,
                    is_garbled=_is_garbled(text),
                    n_images=_count_images(page),
                )
            )

    return records


# ---------------------------------------------------------------------------
# Public pipeline function


def parse_slides(cfg) -> int:
    """Parse every slide PDF in the manifest and write SlideRecord JSONL."""
    del cfg  # Paths are managed centrally through src.config.PATHS.

    manifest_path = Path(PATHS["manifest"])
    raw_root = Path(PATHS["root"]) / "data" / "raw"
    output_path = Path(PATHS["slides"])

    manifest_rows = read_csv(manifest_path)

    slide_rows = [
        row for row in manifest_rows
        if row.get("doc_type", "").strip().lower() == "slides"
    ]

    if not slide_rows:
        raise ValueError(
            f"No slide rows found in manifest: {manifest_path}"
        )

    all_records = []

    for row in slide_rows:
        relative_file = row.get("file", "").strip()

        if not relative_file:
            raise ValueError(
                f"Empty file path for source_id={row.get('source_id')}"
            )

        pdf_path = raw_root / Path(relative_file)

        records = _parse_slide_pdf(
            pdf_path=pdf_path,
            source_id=row["source_id"],
            course_code=row["course_code"],
            sessions=_parse_sessions(row.get("sessions", "")),
        )

        all_records.extend(records)

    return write_jsonl(output_path, all_records)
