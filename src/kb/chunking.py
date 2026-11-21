# Cut the course material into chunks: the pieces of text that are turned into vectors, returned by
# the search and read by the LLM. It reads the syllabus sections (parse_syllabi.py), the slides
# (parse_slides.py), the image text (src/vision) and the slide tables (slide_tables.py), and writes:
#   data/processed/chunks__<kb_sig>.jsonl   one Chunk per piece (format in src/schemas.py)
#   data/processed/parents__<kb_sig>.jsonl  the whole section or slide each chunk was cut from,
#                                           saved for parent-child retrieval but not used by the
#                                           answer phase: the LLM reads the chunks themselves
# The chunks file is read next by dedup.py, which marks near-duplicates, and by build_indexes.py, which
# stores the rest in ChromaDB and BM25.
# kb_sig is a code made from the knowledge-base settings, so each configuration writes its own files.
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from src.config import PATHS, chunks_path, parents_path
# The text helpers: readable syllabus tables, the sentence-based cutting, one summary per course,
# the course names, the real section headings and the filter for empty pieces.
from src.kb.syllabus_text import (
    course_names, course_summary_chunks, is_noise, section_heading, section_text, session_numbers,
    split_into_chunks, with_course,
)
from src.schemas import Chunk, Parent
from src.utils.ids import chunk_id
from src.utils.io import read_jsonl, write_jsonl


# ----------------------------------------------------------------------
# Text of the images and tables of each slide

# The generic opening the vision model often puts at the start of a description, with the punctuation after it. It carries no information, so it is removed
# and the description starts with its content.
_DESCRIPTION_FILLER_PREFIX = re.compile(
    r"^\s*(?:the\s+image\s+(?:shows|is|contains|illustrates)|"
    r"the\s+chart\s+(?:shows|is|displays))\b"
    r"(?:\s*[,.:;!?…]+)?(?:\s*[–—]\s*)?\s*",
    re.IGNORECASE,
)


# Remove that opening from a description; the rest of the text is not changed.
def _strip_description_filler(description: str) -> str:
    # match() only looks at the start of the text, so the same words later in the description are kept.
    match = _DESCRIPTION_FILLER_PREFIX.match(description)
    # No generic opening: the description is returned as it is.
    if not match:
        return description
    # Keep everything after the opening, without the spaces left in front.
    return description[match.end():].lstrip()


# Group the text of the images by slide: slide_id -> list of image texts. Each image text is written
# as "[Image: <class>] OCR: ... Description: ...". use_ocr and use_captions come from the config
# (kb.use_ocr and kb.use_captions), so a configuration can leave out the OCR text, the descriptions or both.
    
def _load_image_text(use_ocr: bool, use_captions: bool) -> dict:
    # defaultdict(list) creates an empty list the first time a slide appears, so the texts of
    # several images of the same slide can be added to it one after another.
    result = defaultdict(list)

    # Nothing to add when both are switched off (the first version of the system had no image text).
    if not use_ocr and not use_captions:
        return result

    image_path = PATHS["image_texts"]

    # Without the image file the chunks are still built, only without image text.
    if not image_path.exists():
        print(f"Warning: image text file not found: {image_path}")
        return result

    # One row per image: its slide, its class, its OCR text and its description.
    images = read_jsonl(image_path)

    for image in images:
        # Photos and decorations (logos, icons, branding), as classified by the vision model, are skipped:
        # their OCR text, such as the university name, would only add noise to the chunks.
        if image.get("image_class", "") in {"photo", "decoration"}:
            continue

        # The pieces of text of this image: its OCR text, its description, or both.
        parts = []

        # The words read inside the image by EasyOCR.
        if use_ocr and image.get("ocr_text", "").strip():
            parts.append(f"OCR: {image['ocr_text'].strip()}")

        # What the vision model described, without its generic opening.
        if use_captions and image.get("description", "").strip():
            description = _strip_description_filler(image["description"])
            if description.strip():
                parts.append(f"Description: {description.strip()}")

        if parts:
            # The "[Image: <class>]" tag marks where the image text starts. The app looks for it
            # (app_components/sources.py) to label a source as coming from an image and show that text.
            # An image without a class (not processed by the vision model) is labelled "image".
            image_class = image.get("image_class") or "image"
            # The image text is added to the list of its slide, so build_chunks() can find it by slide_id.
            result[image["slide_id"]].append(f"[Image: {image_class}] " + " ".join(parts))

    return result


# The tables of each slide, written row by row by slide_tables.py where slide_id points to a list of table texts.
# Returns an empty dictionary when that step has not been run.
def _load_slide_tables() -> dict[str, list[str]]:
    path = Path(PATHS["interim"]) / "slide_tables.jsonl"
    # Without the file the chunks are still built, only without the slide tables.
    if not path.exists():
        print("Slide tables not found: run the slide_tables step first.")
        return {}
    # One entry per slide that has a table, so build_chunks() can find its tables by slide_id.
    return {row["slide_id"]: row["tables"] for row in read_jsonl(path)}


# ----------------------------------------------------------------------
# Build the chunks

# Build every Chunk and Parent and write them to disk. Returns the number of chunks.
# Called by the build pipeline (src/pipeline.py) after the course table and before dedup.
def build_chunks(cfg) -> int:

    # The settings of the knowledge base (the kb group of the config).
    settings = cfg.kb
    # Maximum characters per chunk, and how many characters of the end of a chunk are repeated
    # at the start of the next one.
    chunk_size = int(settings.get("chunk_size", 800))
    overlap = int(settings.get("chunk_overlap", 120))
    # Which material goes into the knowledge base; the first version of the system used only the syllabi.
    sources = settings.get("sources", ["syllabi", "slides"])

    # Whether the OCR text and the image descriptions are added to the slides (both true in base.yaml).
    use_ocr = bool(settings.get("use_ocr", False))
    use_captions =
