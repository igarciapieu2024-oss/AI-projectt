"""Build chunks and parents from parsed documents and optional image text."""

from __future__ import annotations

import re
from collections import defaultdict
from urllib.parse import urlsplit

from src.config import PATHS, chunks_path, parents_path
from src.kb.syllabus_text import (  # readable tables, one summary per course, course names, real headings
    course_names, course_summary_chunks, is_noise, section_heading, section_text, session_numbers,
    split_into_chunks, with_course,
)
from src.schemas import Chunk, Parent
from src.utils.ids import chunk_id
from src.utils.io import read_jsonl, write_jsonl


_DESCRIPTION_FILLER_PREFIX = re.compile(
    r"^\s*(?:the\s+image\s+(?:shows|is|contains|illustrates)|"
    r"the\s+chart\s+(?:shows|is|displays))\b"
    r"(?:\s*[,.:;!?…]+)?(?:\s*[–—]\s*)?\s*",
    re.IGNORECASE,
)
_URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\[\]{}]+")
_URL_TRAILING_PUNCTUATION = ".,;:!?)]}\"'"


def _strip_description_filler(description: str) -> str:
    """Remove a generic description opening without altering later wording."""
    match = _DESCRIPTION_FILLER_PREFIX.match(description)
    if not match:
        return description
    return description[match.end():].lstrip()


def _shorten_urls(text: str) -> str:
    """Replace retrieval-facing URLs with their host while retaining surrounding punctuation."""
    def replace(match: re.Match[str]) -> str:
        value = match.group(0)
        url = value.rstrip(_URL_TRAILING_PUNCTUATION)
        punctuation = value[len(url):]
        parsed = urlsplit(url if "://" in url else f"//{url}")
        host = (parsed.hostname or "").lower()
        if not host:
            return value
        domain = host[4:] if host.startswith("www.") else host
        return f"source: {domain}{punctuation}"

    return _URL_PATTERN.sub(replace, text)


def _load_image_text(use_ocr: bool, use_captions: bool) -> dict:
    """Group OCR text and image descriptions by slide ID."""
    result = defaultdict(list)

    if not use_ocr and not use_captions:
        return result

    image_path = PATHS["image_texts"]

    if not image_path.exists():
        print(f"Warning: image text file not found: {image_path}")
        return result

    images = read_jsonl(image_path)

    for image in images:
        # photos and decorations (logos, icons, branding), as classified by the vision model,
        # are skipped: their OCR text ("ie UNIVERSITY"...) would only add noise to the chunks
        if image.get("image_class", "") in {"photo", "decoration"}:
            continue

        parts = []

        if use_ocr and image.get("ocr_text", "").strip():
            parts.append(f"OCR: {image['ocr_text'].strip()}")  # the text read inside the image

        if use_captions and image.get("description", "").strip():
            description = _strip_description_filler(image["description"])
            if description.strip():
                parts.append(f"Description: {description.strip()}")  # what the vision model saw

        if parts:
            # same format as the plan: "[Image: diagram] OCR: ... Description: ..."
            # the app looks for "[Image: <class>]" to show the "From a diagram" tag and the image text box
            image_class = image.get("image_class") or "image"
            result[image["slide_id"]].append(f"[Image: {image_class}] " + " ".join(parts))

    return result


def build_chunks(cfg) -> int:
    """Build Chunk and Parent records; return the number of chunks."""

    settings = cfg.kb
    chunk_size = int(settings.get("chunk_size", 800))
    overlap = int(settings.get("chunk_overlap", 120))
    sources = settings.get("sources", ["syllabi", "slides"])

    use_ocr = bool(settings.get("use_ocr", False))
    use_captions = bool(settings.get("use_captions", False))

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    overlap = max(0, min(overlap, chunk_size - 1))

    image_text = _load_image_text(use_ocr, use_captions)
    names = course_names()  # "mgmt_control" -> "Management Control", added to the searched text of every chunk

    chunks = []
    parents = []

    # Process syllabus sections
    if "syllabi" in sources:
        sections = read_jsonl(PATHS["syllabus_sections"])

        for section in sections:
            record_id = section["section_id"]
            text = section_text(section)  # body + tables written as readable lines ("- Final Exam: 40%. ...")

            outline_path = f"Syllabus › {section_heading(section)}"  # the real heading, not "OTHER"
            parent_id = f"parent__{record_id}"

            parents.append(
                Parent(
                    parent_id=parent_id,
                    outline_path=outline_path,
                    covers=[record_id],
                    text=text,
                )
            )

            # the programme is cut at "SESSION N" so each chunk holds whole sessions and knows their numbers
            is_program = section.get("section_name") == "PROGRAM"
            pieces = split_into_chunks(text, chunk_size, overlap, by_session=is_program)
            piece_sessions: list[int] = []
            for i, piece in enumerate(pieces):
                piece_sessions = session_numbers(piece, piece_sessions) if is_program else []
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id("section", record_id, i),
                        kb_sig=cfg.kb_sig,
                        strategy="section",
                        doc_type="syllabus_section",
                        course_code=section["course_code"],
                        sessions=piece_sessions,
                        source_id=section["source_id"],
                        covers=[record_id],
                        parent_id=parent_id,
                        outline_path=outline_path,
                        text=piece,
                        embed_text=with_course(names.get(section["course_code"], ""), outline_path, piece),
                        has_image_text=False,
                        duplicate_of=None,
                    )
                )

        # one clean summary chunk per course from the course table (final exam weight, components, attendance...)
        if settings.get("course_rows", False):
            summary_chunks, summary_parents = course_summary_chunks(cfg.kb_sig)
            if not summary_chunks:
                print("Course table not found: no course summary chunks (run the course_table step first).")
            chunks.extend(summary_chunks)
            parents.extend(summary_parents)

    # Process slides and their associated image text
    if "slides" in sources:
        slides = read_jsonl(PATHS["slides"])

        for slide in slides:
            record_id = slide["slide_id"]
            original_text = slide.get("text", "")

            image_parts = image_text.get(record_id, [])
            has_image_text = bool(image_parts)

            text = original_text
            if image_parts:
                text += "\n\n" + "\n\n".join(image_parts)

            if not text.strip():
                continue

            sessions = slide.get("sessions", [])
            session_label = (
                f"Session {', '.join(map(str, sessions))}"
                if sessions
                else "Slides"
            )

            title = slide.get("title", "").strip()
            outline_path = (
                f"{session_label} › {title}"
                if title
                else session_label
            )

            parent_id = f"parent__{record_id}"

            parents.append(
                Parent(
                    parent_id=parent_id,
                    outline_path=outline_path,
                    covers=[record_id],
                    text=text,
                )
            )

            for i, piece in enumerate(split_into_chunks(text, chunk_size, overlap)):  # cut at sentence ends
                if is_noise(piece, has_image_text):
                    continue  # "THANK YOU", a page number...
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id("slide", record_id, i),
                        kb_sig=cfg.kb_sig,
                        strategy="slide",
                        doc_type="slide",
                        course_code=slide["course_code"],
                        sessions=sessions,
                        source_id=slide["source_id"],
                        covers=[record_id],
                        parent_id=parent_id,
                        outline_path=outline_path,
                        text=piece,
                        embed_text=with_course(names.get(slide["course_code"], ""), outline_path, piece),
                        has_image_text=has_image_text,
                        duplicate_of=None,
                    )
                )

    # Both dense and BM25 indexes use embed_text. Keep the original URL in text for traceability.
    for chunk in chunks:
        chunk.embed_text = _shorten_urls(chunk.embed_text)

    write_jsonl(chunks_path(cfg.kb_sig), chunks)
    write_jsonl(parents_path(cfg.kb_sig), parents)

    print(f"Created {len(chunks)} chunks and {len(parents)} parents.")
    print(f"OCR enabled: {use_ocr}")
    print(f"Captions enabled: {use_captions}")
    print(f"Chunks: {chunks_path(cfg.kb_sig)}")
    print(f"Parents: {parents_path(cfg.kb_sig)}")

    return len(chunks)
