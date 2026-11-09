"""Build simple chunks and parent records from parsed slides and syllabi."""

from __future__ import annotations

from src.schemas import Chunk, Parent
from src.utils.ids import chunk_id
from src.utils.io import read_jsonl, write_jsonl
from src.config import PATHS, chunks_path, parents_path


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks, keeping word boundaries where possible."""
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


def build_chunks(cfg) -> int:
    """Build Chunk and Parent records; return the number of chunks."""

    settings = cfg.kb
    chunk_size = int(settings.get("chunk_size", 800))
    overlap = int(settings.get("chunk_overlap", 120))
    sources = settings.get("sources", ["syllabi", "slides"])

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    overlap = max(0, min(overlap, chunk_size - 1))

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

    # Process slides
    if "slides" in sources:
        slides = read_jsonl(PATHS["slides"])

        for slide in slides:
            record_id = slide["slide_id"]
            text = slide.get("text", "")

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
                        has_image_text=False,
                        duplicate_of=None,
                    )
                )

    write_jsonl(chunks_path(cfg.kb_sig), chunks)
    write_jsonl(parents_path(cfg.kb_sig), parents)

    print(f"Created {len(chunks)} chunks and {len(parents)} parents.")
    print(f"Chunks: {chunks_path(cfg.kb_sig)}")
    print(f"Parents: {parents_path(cfg.kb_sig)}")

    return len(chunks)
