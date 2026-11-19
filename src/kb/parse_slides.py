# Parse the slide PDFs listed in the manifest into SlideRecord objects.
# Output: data/interim/slides.jsonl, with one record per PDF page.

from __future__ import annotations

import re
from pathlib import Path

import pymupdf  # PDF library used to read the text, positions and fonts of each page

from src.config import PATHS
from src.kb.clean import remove_repeated_footers
from src.schemas import SlideRecord
from src.utils.ids import slide_id
from src.utils.io import read_csv, write_jsonl


# ---------------------------------------------------------------------------
# Text processing helpers


# Replace any run of spaces, tabs or line breaks with a single space.
def _clean_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# Return every text line of a page with its position, font size and bold flag.
# A PDF does not store sentences, it stores pieces of text ("spans") placed at
# coordinates on the page. We keep the position and the font of each line so we
# can rebuild the reading order and detect the title afterwards.
def _extract_lines(page: pymupdf.Page) -> list[dict]:
    result = []
    page_dict = page.get_text("dict")  # the page content as text blocks, each with lines, each with spans (pieces of text with the same font)

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:  # type 0 is text; other blocks (images) are skipped
            continue

        for line in block.get("lines", []):
            spans = line.get("spans", [])
            # Join the spans of the line into one string.
            text = _clean_line(
                "".join(span.get("text", "") for span in spans)
            )

            if not text:  # empty lines carry no information
                continue

            # A line can mix several font sizes; we keep the largest one,
            # ignoring spans that only contain spaces.
            font_sizes = [
                float(span.get("size", 0))
                for span in spans
                if span.get("text", "").strip()
            ]

            result.append({
                "text": text,
                "x": float(line["bbox"][0]),  # horizontal position (left edge)
                "y": float(line["bbox"][1]),  # vertical position (top edge)
                "font_size": max(font_sizes, default=0.0),
                # The line counts as bold if any of its spans uses a bold font.
                "bold": any(
                    "bold" in span.get("font", "").lower()
                    for span in spans
                ),
            })

    # Reading order: top to bottom, and left to right for lines at the same height.
    return sorted(result, key=lambda item: (item["y"], item["x"]))


# Join the lines, already in reading order, into the text of the slide.
def _extract_text(lines: list[dict]) -> str:
    return "\n".join(item["text"] for item in lines).strip()


# Choose the slide title among the lines in the top 40% of the page.
def _find_title(lines: list[dict], page_height: float) -> str:
    # Candidates: lines near the top, with a reasonable length,
    # that are not just a number (page numbers are never titles).
    candidates = [
        item for item in lines
        if item["y"] <= page_height * 0.40
        and 2 <= len(item["text"]) <= 180
        and not re.fullmatch(r"\d+", item["text"])
    ]

    if not candidates:  # no line qualifies, so the slide has no title
        return ""

    # Sort so the best candidate comes first: largest font, then bold text,
    # then the line closest to the top (and to the left) to break ties.
    candidates.sort(
        key=lambda item: (
            -item["font_size"],
            not item["bold"],
            item["y"],
            item["x"],
        )
    )

    return candidates[0]["text"]


# Return the lines that look like headings inside the slide.
def _find_outline(lines: list[dict], title: str) -> list[str]:
    if not lines:
        return []

    # Median font size of the slide, used as the reference for normal text.
    sizes = sorted(
        item["font_size"] for item in lines if item["font_size"] > 0
    )
    median_size = sizes[len(sizes) // 2] if sizes else 0

    outline = []

    for item in lines:
        text = item["text"]

        # The title is already stored separately, and long lines are body text.
        if text == title or len(text) > 150:
            continue

        # Numbered headings such as "1.", "2.3" or "A)" followed by text.
        numbered = bool(
            re.match(
                r"^(?:\d+(?:\.\d+)*[.)]?|[A-Z][.)])\s+\S",
                text,
            )
        )

        # Headings by typography: font at least 20% larger than the median.
        typography_heading = (
            item["font_size"] >= median_size * 1.20
            and len(text) <= 110
        )

        if numbered or typography_heading:
            if text not in outline:  # avoid repeating the same heading
                outline.append(text)

    return outline


# Detect agenda or contents slides by looking for common keywords
# in the title and the beginning of the text.
def _is_agenda(title: str, text: str) -> bool:
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


# Flag slides whose text was not extracted correctly.
def _is_garbled(text: str) -> bool:
    if not text:
        return False

    # Characters that appear when a font cannot be decoded: the Unicode
    # replacement character and invisible control characters.
    replacement_count = text.count("\ufffd")
    suspicious_count = sum(
        1 for char in text
        if ord(char) < 32 and char not in "\n\r\t"
    )

    total = max(len(text), 1)  # avoids dividing by zero

    # Garbled if more than 1% of the characters are of either kind.
    return (
        replacement_count / total > 0.01
        or suspicious_count / total > 0.01
    )


# Count the images on a page (no OCR is done here).
def _count_images(page: pymupdf.Page) -> int:
    try:
        return len(page.get_image_info(xrefs=True))
    except (AttributeError, RuntimeError):
        # Older PyMuPDF versions do not have get_image_info.
        return len(page.get_images(full=True))



# Parse the sessions column of the manifest
# into a list of integers.
def _parse_sessions(value: str) -> list[int]:
    if not value or not value.strip():  # a syllabus row has no sessions
        return []

    # Remove optional square brackets and split on commas or semicolons.
    cleaned = value.strip().strip("[]")
    try:
        return [
            int(item.strip())
            for item in re.split(r"[,;]", cleaned)
            if item.strip()
        ]
    except ValueError as exc:
        # A value that is not a number means the manifest row is wrong.
        raise ValueError(
            f"Invalid session value in manifest: {value!r}"
        ) from exc


# ---------------------------------------------------------------------------
# PDF parsing


# Read one slide deck and return one SlideRecord per page.
# The course and sessions come from the manifest; the rest is read from each page.
def _parse_slide_pdf(
    pdf_path: str | Path,
    source_id: str,
    course_code: str,
    sessions: list[int],
) -> list[SlideRecord]:
    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():  # stop early with a clear message if the PDF is missing
        raise FileNotFoundError(f"Slide PDF not found: {pdf_path}")

    records = []

    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            page_number = page_index + 1  # page numbers start at 1, as in the PDF

            # Text of the page in reading order.
            lines = _extract_lines(page)
            text = _extract_text(lines)

            # Title and headings, detected from position and font size.
            title = _find_title(lines, float(page.rect.height))
            outline = _find_outline(lines, title)

            records.append(
                SlideRecord(
                    # The id depends only on the document and the page number,
                    # so it stays the same every time the slides are parsed.
                    slide_id=slide_id(source_id, page_number),
                    source_id=source_id,
                    course_code=course_code,
                    sessions=sessions,
                    page=page_number,
                    title=title,
                    outline=outline,
                    text=text,
                    # Quality flags: agenda slides, slides with almost no text
                    # (under 40 characters) and slides with garbled text.
                    is_agenda=_is_agenda(title, text),
                    is_low_text=len(text.strip()) < 40,
                    is_garbled=_is_garbled(text),
                    n_images=_count_images(page),
                )
            )

    return records


# ---------------------------------------------------------------------------
# Public pipeline function


# Parse every slide deck in the manifest, write data/interim/slides.jsonl
# and return the number of records written.
def parse_slides(cfg) -> int:
    del cfg  # not needed: all paths come from src.config.PATHS

    manifest_path = Path(PATHS["manifest"])
    raw_root = Path(PATHS["root"]) / "data" / "raw"  # folder with the PDFs
    output_path = Path(PATHS["slides"])

    manifest_rows = read_csv(manifest_path)

    # Keep only the slide decks; the syllabi are parsed separately.
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

        if not relative_file:  # every manifest row must point to a PDF
            raise ValueError(
                f"Empty file path for source_id={row.get('source_id')}"
            )

        pdf_path = raw_root / Path(relative_file)

        # One record per page, with the course and sessions of this manifest row.
        records = _parse_slide_pdf(
            pdf_path=pdf_path,
            source_id=row["source_id"],
            course_code=row["course_code"],
            sessions=_parse_sessions(row.get("sessions", "")),
        )

        all_records.extend(records)

    # Footer lines repeated across many slides (see clean.py) are removed
    # before saving, so they do not end up in the searchable text.
    all_records = remove_repeated_footers(all_records)
    return write_jsonl(output_path, all_records)
