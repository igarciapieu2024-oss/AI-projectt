"""Build chunks and parents from parsed documents and optional image text."""

from __future__ import annotations

from collections import defaultdict

from src.config import PATHS, chunks_path, parents_path
from src.schemas import Chunk, Parent
from src.utils.ids import chunk_id
from src.utils.io import read_jsonl, write_jsonl


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks, keeping word boundaries."""
    text = " ".join((text or "").split())

    if not text:
        return []

    if len(text) <= size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


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
        parts = []

        if use_ocr and image.get("ocr_text", "").strip():
            parts.append(f"Image OCR: {image['ocr_text'].strip()}")

        if use_captions and image.get("description", "").strip():
            parts.append(f"Image description: {image['description'].strip()}")

        if parts:
            result[image["slide_id"]].extend(parts)

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

    chunks = []
    parents = []

    # Process syllabus sections
    if "syllabi" in sources:
        sections = read_jsonl(PATHS["syllabus_sections"])

        for section in sections:
            record_id = section["section_id"]
            text = section.get("text", "")

            tables = section.get("tables", [])
            if tables:
                text += "\n\n" + "\n\n".join(tables)

            outline_path = f"Syllabus › {section.get('section_name', 'Section')}"
            parent_id = f"parent__{record_id}"

            parents.append(
                Parent(
                    parent_id=parent_id,
                    outline_path=outline_path,
                    covers=[record_id],
                    text=text,
                )
            )

            for i, piece in enumerate(_split_text(text, chunk_size, overlap)):
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id("section", record_id, i),
                        kb_sig=cfg.kb_sig,
                        strategy="section",
                        doc_type="syllabus_section",
                        course_code=section["course_code"],
                        sessions=[],
                        source_id=section["source_id"],
                        covers=[record_id],
                        parent_id=parent_id,
                        outline_path=outline_path,
                        text=piece,
                        embed_text=f"{outline_path}\n{piece}",
                        has_image_text=False,
                        duplicate_of=None,
                    )
                )

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

            for i, piece in enumerate(_split_text(text, chunk_size, overlap)):
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
                        embed_text=f"{outline_path}\n{piece}",
                        has_image_text=has_image_text,
                        duplicate_of=None,
                    )
                )

    write_jsonl(chunks_path(cfg.kb_sig), chunks)
    write_jsonl(parents_path(cfg.kb_sig), parents)

    print(f"Created {len(chunks)} chunks and {len(parents)} parents.")
    print(f"OCR enabled: {use_ocr}")
    print(f"Captions enabled: {use_captions}")
    print(f"Chunks: {chunks_path(cfg.kb_sig)}")
    print(f"Parents: {parents_path(cfg.kb_sig)}")

    return len(chunks)
