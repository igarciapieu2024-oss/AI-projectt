# Read the slide PDFs listed in the manifest and turn every page into a SlideRecord. The output is
# data/interim/slides.jsonl, with one record per PDF page. The chunking step (src/kb/chunking.py) builds the
# searchable chunks from these records.

from __future__ import annotations

# Regular expressions, to clean spaces and recognise headings.
import re
from pathlib import Path

import pymupdf  # PDF library used to read the text, positions and fonts of each page

# The folders of the project, the footer removal (clean.py), the format of a slide record, the builder of slide ids
# and the CSV reader and JSONL writer.
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


# Return every text line of a page with its position, its font size and whether it's bold. A PDF stores pieces of text
# ("spans") placed at coordinates on the page. We need the position to rebuild the reading order, and the font size
# and weight to recognise the title.
def _extract_lines(page: pymupdf.Page) -> list[dict]:
    # The lines of the page, filled below.
    result = []
    # The content of the page as text blocks, each with its lines, and each line with its spans (pieces of text that
    # share the same font).
    page_dict = page.get_text("dict")

    # Go through the blocks of the page.
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:  # type 0 is text; image blocks are skipped
            continue

        for line in block.get("lines", []):
            spans = line.get("spans", [])
            # Join the spans of the line into one string.
            text = _clean_line(
                "".join(span.get("text", "") for span in spans)
            )

            if not text:  # empty lines carry no information
                continue

            # A line can mix several font sizes; we keep the largest one, ignoring spans that only contain spaces.
            font_sizes = [
                float(span.get("size", 0))
                for span in spans
                if span.get("text", "").strip()
            ]

            # One entry per line: its text, where it is on the page and how it is written.
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


# Join the lines, already sorted into reading order, into the text of the slide. We need the sorting because a PDF
# doesn't store its text in the order it's read.
def _extract_text(lines: list[dict]) -> str:
    return "\n".join(item["text"] for item in lines).strip()


# Choose the title of the slide among the lines in the top 40% of the page, where slide titles are placed. The title
# becomes part of the position of each chunk ("Session 9 › title") in the chunking step.
def _find_title(lines: list[dict], page_height: float) -> str:
    # The candidates: lines near the top, with a reasonable length, that aren't just a number (page numbers are never
    # titles).
    candidates = [
        item for item in lines
        if item["y"] <= page_height * 0.40
        and 2 <= len(item["text"]) <= 180
        and not re.fullmatch(r"\d+", item["text"])
    ]

    if not candidates:  # no line qualifies, so the slide has no title
        return ""

    # We sort them so the best candidate comes first: the largest font, then bold text, then the line closest to the
    # top (and to the left) to break ties.
    candidates.sort(
        key=lambda item: (
            -item["font_size"],
            not item["bold"],
            item["y"],
            item["x"],
        )
    )

    return candidates[0]["text"]


# Return the lines that look like headings inside the slide. We store them in the outline field of the record.
def _find_outline(lines: list[dict], title: str) -> list[str]:
    # A slide without lines has no headings.
    if not lines:
        return []

    # The median font size of the slide, which we take as the size of normal body text.
    sizes = sorted(
        item["font_size"] for item in lines if item["font_size"] > 0
    )
    median_size = sizes[len(sizes) // 2] if sizes else 0

    # The headings we find, in reading order.
    outline = []

    for item in lines:
        text = item["text"]

        # The title is already stored separately, and long lines are body text.
        if text == title or len(text) > 150:
            continue

        # Numbered headings followed by text, such as "2.1 Topic".
        numbered = bool(
            re.match(
                r"^(?:\d+(?:\.\d+)*[.)]?|[A-Z][.)])\s+\S",
                text,
            )
        )

        # Headings by their typography: a font at least 20% larger than the median.
        typography_heading = (
            item["font_size"] >= median_size * 1.20
            and len(text) <= 110
        )

        if numbered or typography_heading:
            if text not in outline:  # avoid repeating the same heading
                outline.append(text)

    return outline


# Detect agenda or contents slides by looking for common keywords in the title and in the beginning of the text.
def _is_agenda(title: str, text: str) -> bool:
    # The title and the first 1,500 characters of the text, in lower case.
    sample = f"{title}\n{text[:1500]}".lower()

    # The words that usually appear on an agenda or contents slide.
    patterns = (
        r"\bagenda\b",
        r"\btable of contents\b",
        r"\bcontents\b",
        r"\boutline\b",
        r"\btoday'?s topics\b",
        r"\blearning objectives\b",
        r"\broadmap\b",
    )

    # It's an agenda slide if any of them appears.
    return any(re.search(pattern, sample) for pattern in patterns)


# Flag the slides whose text wasn't read correctly, which happens when a PDF uses a font that PyMuPDF can't decode.
def _is_garbled(text: str) -> bool:
    # An empty text isn't garbled, just empty.
    if not text:
        return False

    # The characters that appear when a font can't be decoded: the Unicode replacement character and invisible control
    # characters.
    replacement_count = text.count("\ufffd")
    suspicious_count = sum(
        1 for char in text
        if ord(char) < 32 and char not in "\n\r\t"
    )

    total = max(len(text), 1)  # avoids dividing by zero

    # It's garbled if more than 1% of the characters are of either kind.
    return (
        replacement_count / total > 0.01
        or suspicious_count / total > 0.01
    )


# Count the images on a page. We only store the count here; the images themselves are extracted and read in the image
# step (src/vision/).
def _count_images(page: pymupdf.Page) -> int:
    # Newer PyMuPDF versions list the images that are really drawn on the page.
    try:
        return len(page.get_image_info(xrefs=True))
    except (AttributeError, RuntimeError):
        # Older PyMuPDF versions don't have get_image_info.
        return len(page.get_images(full=True))



# Turn the sessions column of the manifest into a list of whole numbers.
def _parse_sessions(value: str) -> list[int]:
    if not value or not value.strip():  # a syllabus row has no sessions
        return []

    # Remove the optional square brackets and split on commas or semicolons.
    cleaned = value.strip().strip("[]")
    try:
        return [
            int(item.strip())
            for item in re.split(r"[,;]", cleaned)
            if item.strip()
        ]
    except ValueError as exc:
        # A value that isn't a number means the manifest row is wrong.
        raise ValueError(
            f"Invalid session value in manifest: {value!r}"
        ) from exc


# ---------------------------------------------------------------------------
# PDF parsing


# Read one slide deck and return one SlideRecord per page. The course and the sessions come from the manifest; the
# rest is read from each page.
def _parse_slide_pdf(
    pdf_path: str | Path,
    source_id: str,
    course_code: str,
    sessions: list[int],
) -> list[SlideRecord]:
    # The path of the PDF.
    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():  # stop early with a clear message if the PDF is missing
        raise FileNotFoundError(f"Slide PDF not found: {pdf_path}")

    # One record per page, filled below.
    records = []

    # Open the PDF; the `with` block closes it when we finish.
    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            page_number = page_index + 1  # page numbers start at 1, as in the PDF

            # The text of the page in reading order.
            lines = _extract_lines(page)
            text = _extract_text(lines)

            # The title and the headings, found from the position and the font size.
            title = _find_title(lines, float(page.rect.height))
            outline = _find_outline(lines, title)

            records.append(
                SlideRecord(
                    # The id only depends on the document and the page number, so it stays the same after every
                    # rebuild. The evaluation questions use these ids to name their correct source.
                    slide_id=slide_id(source_id, page_number),
                    source_id=source_id,
                    course_code=course_code,
                    sessions=sessions,
                    page=page_number,
                    title=title,
                    outline=outline,
                    text=text,
                    # The quality flags. Slides with almost no text (under 40 characters) or with garbled text are
                    # rendered as a whole image in the image step (src/vision/extract.py), so OCR can read what we
                    # missed here. is_agenda marks agenda and contents slides.
                    is_agenda=_is_agenda(title, text),
                    is_low_text=len(text.strip()) < 40,
                    is_garbled=_is_garbled(text),
                    n_images=_count_images(page),
                )
            )

    return records


# ---------------------------------------------------------------------------
# Public pipeline function


# Read every slide deck in the manifest, write data/interim/slides.jsonl and return the number of records written.
# src/pipeline.py calls it.
def parse_slides(cfg) -> int:
    del cfg  # not needed: all paths come from src.config.PATHS

    # The manifest, the folder with the PDFs and the file we write.
    manifest_path = Path(PATHS["manifest"])
    raw_root = Path(PATHS["root"]) / "data" / "raw"  # folder with the PDFs
    output_path = Path(PATHS["slides"])

    # Every row of the manifest.
    manifest_rows = read_csv(manifest_path)

    # We keep only the slide decks; the syllabi are read by parse_syllabi.py.
    slide_rows = [
        row for row in manifest_rows
        if row.get("doc_type", "").strip().lower() == "slides"
    ]

    # A manifest without slide decks is a mistake.
    if not slide_rows:
        raise ValueError(
            f"No slide rows found in manifest: {manifest_path}"
        )

    # The records of all the decks together.
    all_records = []

    for row in slide_rows:
        # The path of the PDF, written in the manifest relative to data/raw.
        relative_file = row.get("file", "").strip()

        if not relative_file:  # every manifest row must point to a PDF
            raise ValueError(
                f"Empty file path for source_id={row.get('source_id')}"
            )

        pdf_path = raw_root / Path(relative_file)

        # One record per page, with the course and the sessions of this manifest row.
        records = _parse_slide_pdf(
            pdf_path=pdf_path,
            source_id=row["source_id"],
            course_code=row["course_code"],
            sessions=_parse_sessions(row.get("sessions", "")),
        )

        # Add the pages of this deck to the rest.
        all_records.extend(records)

    # Footer lines repeated on many slides of the same course, such as the lecturer's name, say nothing about the
    # slide itself, so we remove them (see clean.py) before saving the records.
    all_records = remove_repeated_footers(all_records)
    # Write all the records and return how many there are.
    return write_jsonl(output_path, all_records)
