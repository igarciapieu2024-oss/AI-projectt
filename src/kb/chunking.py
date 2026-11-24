# This file cuts the course material into chunks: the pieces of text that we turn into vectors, that the search gives
# back and that the LLM reads. It reads the syllabus sections (parse_syllabi.py), the slides (parse_slides.py), the
# image text (src/vision) and the slide tables (slide_tables.py), and writes two files:
#   data/processed/chunks__<kb_sig>.jsonl   one Chunk per piece (format in src/schemas.py)
#   data/processed/parents__<kb_sig>.jsonl  the whole section or slide each chunk was cut from. We save them for
# parent-child retrieval, but the answer phase doesn't use them: the LLM reads the chunks themselves.
# The chunks file is read next by dedup.py, which marks near-duplicates, and by build_indexes.py, which stores the
# rest in ChromaDB and BM25.
# kb_sig is a code made from the knowledge-base settings, so each configuration writes its own files.
from __future__ import annotations

# Regular expressions, to recognise the generic opening of an image description.
import re
# defaultdict gives us an empty list the first time we use a new key.
from collections import defaultdict
from pathlib import Path

# Where the files are read from and written to.
from src.config import PATHS, chunks_path, parents_path
# The text helpers: readable syllabus tables, the sentence-based cutting, one summary per course,
# the course names, the real section headings and the filter for empty pieces.
from src.kb.syllabus_text import (
    course_names, course_summary_chunks, is_noise, section_heading, section_text, session_numbers,
    split_into_chunks, with_course,
)
# The formats of a chunk and of its parent, the function that builds chunk ids, and the JSONL reader and writer.
from src.schemas import Chunk, Parent
from src.utils.ids import chunk_id
from src.utils.io import read_jsonl, write_jsonl


# ----------------------------------------------------------------------
# Text of the images and tables of each slide

# The vision model often starts a description with a generic opening ("The image shows", "The chart displays"...),
# followed by some punctuation. It carries no information, so we remove it and the description starts with its
# content.
_DESCRIPTION_FILLER_PREFIX = re.compile(
    r"^\s*(?:the\s+image\s+(?:shows|is|contains|illustrates)|"
    r"the\s+chart\s+(?:shows|is|displays))\b"
    r"(?:\s*[,.:;!?…]+)?(?:\s*[–—]\s*)?\s*",
    re.IGNORECASE,
)


# Remove that opening from a description; the rest of the text doesn't change.
def _strip_description_filler(description: str) -> str:
    # match() only looks at the start of the text, so the same words later in the description are kept.
    match = _DESCRIPTION_FILLER_PREFIX.match(description)
    # No generic opening: we return the description as it is.
    if not match:
        return description
    # We keep everything after the opening, without the spaces left in front.
    return description[match.end():].lstrip()


# Group the text of the images by slide: slide_id -> list of image texts. Each image text is written as "[Image:
# <class>] OCR: ... Description: ...". use_ocr and use_captions come from the config (kb.use_ocr and kb.use_captions),
# so a configuration can leave out the OCR text, the descriptions or both.
def _load_image_text(use_ocr: bool, use_captions: bool) -> dict:
    # defaultdict(list) creates an empty list the first time a slide appears, so we can add the texts of several
    # images of the same slide one after another.
    result = defaultdict(list)

    # Nothing to add when both are switched off (our first version had no image text).
    if not use_ocr and not use_captions:
        return result

    # The file with the text of every image, written by the image step (src/vision).
    image_path = PATHS["image_texts"]

    # Without the image file we still build the chunks, just without image text.
    if not image_path.exists():
        print(f"Warning: image text file not found: {image_path}")
        return result

    # One row per image: its slide, its class, its OCR text and its description.
    images = read_jsonl(image_path)

    for image in images:
        # We skip photos and decorations (logos, icons, branding), as classified by the vision model: their OCR text,
        # such as the university name, would only add noise to the chunks.
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

        # An image with no usable text adds nothing.
        if parts:
            # The "[Image: <class>]" tag marks where the image text starts. The app looks for it
            # (app_components/sources.py) to label a source as coming from an image and to show that text. An image
            # without a class (not processed by the vision model) is labelled "image".
            image_class = image.get("image_class") or "image"
            # We add the image text to the list of its slide, so build_chunks() can find it by slide_id.
            result[image["slide_id"]].append(f"[Image: {image_class}] " + " ".join(parts))

    return result


# The tables of each slide, written row by row by slide_tables.py: slide_id -> list of table texts. It returns an
# empty dictionary when that step hasn't been run.
def _load_slide_tables() -> dict[str, list[str]]:
    # The file written by slide_tables.py.
    path = Path(PATHS["interim"]) / "slide_tables.jsonl"
    # Without the file we still build the chunks, just without the slide tables.
    if not path.exists():
        print("Slide tables not found: run the slide_tables step first.")
        return {}
    # One entry per slide that has a table, so build_chunks() can find its tables by slide_id.
    return {row["slide_id"]: row["tables"] for row in read_jsonl(path)}


# ----------------------------------------------------------------------
# Build the chunks

# Build every Chunk and Parent and write them to disk. It returns the number of chunks. The build pipeline
# (src/pipeline.py) calls it after the course table and before dedup.
def build_chunks(cfg) -> int:

    # The settings of the knowledge base (the kb group of the config).
    settings = cfg.kb
    # The maximum number of characters per chunk, and how many characters from the end of a chunk we repeat at the
    # start of the next one.
    chunk_size = int(settings.get("chunk_size", 800))
    overlap = int(settings.get("chunk_overlap", 120))
    # Which material goes into the knowledge base; our first version only used the syllabi.
    sources = settings.get("sources", ["syllabi", "slides"])

    # Whether we add the OCR text and the image descriptions to the slides (both true in base.yaml).
    use_ocr = bool(settings.get("use_ocr", False))
    use_captions = bool(settings.get("use_captions", False))

    # A chunk of zero or negative size is a mistake in the config, so we stop before building anything.
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    # The overlap has to be smaller than the chunk itself, otherwise a chunk would only repeat the previous one.
    overlap = max(0, min(overlap, chunk_size - 1))

    # Everything we add to the slides, loaded once before the loop.
    image_text = _load_image_text(use_ocr, use_captions)
    # We only use the slide tables with kb.slide_tables (final_system.yaml); the earlier configurations leave it off,
    # so their knowledge bases are rebuilt exactly as they were.
    table_text = _load_slide_tables() if settings.get("slide_tables", False) else {}
    # course_code -> course name, from the manifest; we put the name in front of the searched text of every chunk.
    names = course_names()

    # The two lists we fill below and write to disk at the end.
    chunks = []
    parents = []

    # ---- Syllabus sections
    if "syllabi" in sources:
        # The sections written by parse_syllabi.py.
        sections = read_jsonl(PATHS["syllabus_sections"])

        for section in sections:
            # The fixed id of the section; we use it to build the ids of its parent and its chunks.
            record_id = section["section_id"]
            # The section body plus its tables rewritten as readable lines (syllabus_text.py).
            text = section_text(section)

            # The position of the chunk in the course, with the real heading of the section instead of the generic
            # label OTHER that parse_syllabi.py gives to sections it doesn't recognise.
            outline_path = f"Syllabus › {section_heading(section)}"
            # The id of the parent is the section id with "parent__" in front.
            parent_id = f"parent__{record_id}"

            # The whole section, which we keep as the parent of its chunks.
            parents.append(
                Parent(
                    parent_id=parent_id,
                    outline_path=outline_path,
                    covers=[record_id],
                    text=text,
                )
            )

            # We cut the programme section at each "SESSION N" instead of at sentence ends, so every chunk holds whole
            # sessions and can store their numbers.
            is_program = section.get("section_name") == "PROGRAM"
            # Cut the section into pieces of at most chunk_size characters (syllabus_text.py).
            pieces = split_into_chunks(text, chunk_size, overlap, by_session=is_program)
            # The session numbers of the previous piece, so a piece that continues a session keeps its number.
            piece_sessions: list[int] = []
            # One chunk per piece; i is the position of the piece inside the section (0, 1, 2...).
            for i, piece in enumerate(pieces):
                # Only programme pieces have session numbers; the other sections aren't tied to a session.
                piece_sessions = session_numbers(piece, piece_sessions) if is_program else []
                chunks.append(
                    Chunk(
                        # Fixed id from the section id and the position of the piece (src/utils/ids.py).
                        chunk_id=chunk_id("section", record_id, i),
                        # The knowledge base this chunk belongs to.
                        kb_sig=cfg.kb_sig,
                        # How the chunk was cut (by section) and what kind of document it comes from.
                        strategy="section",
                        doc_type="syllabus_section",
                        # Course and sessions, saved in ChromaDB and used by the course and session filters.
                        course_code=section["course_code"],
                        sessions=piece_sessions,
                        # The PDF the chunk comes from.
                        source_id=section["source_id"],
                        # covers is the id the evaluation compares with the expected source of a question.
                        covers=[record_id],
                        # The whole section this chunk was cut from.
                        parent_id=parent_id,
                        # The position of the chunk in the course (course part, section heading).
                        outline_path=outline_path,
                        # text is what the LLM reads; embed_text (course name + position + text) is what we turn into
                        # a vector, what the searches look at and what the reranker reads.
                        text=piece,
                        embed_text=with_course(names.get(section["course_code"], ""), outline_path, piece),
                        # Syllabus sections never include image text.
                        has_image_text=False,
                        # Filled later by dedup.py if this chunk is a near-duplicate of another one.
                        duplicate_of=None,
                    )
                )

        # One summary chunk per course, built from the course table (course_table.py), with kb.course_rows.
        if settings.get("course_rows", False):
            # The summary chunks and their parents (syllabus_text.py).
            summary_chunks, summary_parents = course_summary_chunks(cfg.kb_sig)
            # The course table is built earlier in the pipeline; if it's missing, the build goes on without summaries.
            if not summary_chunks:
                print("Course table not found: no course summary chunks (run the course_table step first).")
            # They go into the same lists as the section chunks.
            chunks.extend(summary_chunks)
            parents.extend(summary_parents)

    # ---- Slides, with the text of their tables and images
    if "slides" in sources:
        # The slides written by parse_slides.py, already without repeated footers.
        slides = read_jsonl(PATHS["slides"])

        for slide in slides:
            # The fixed id of the slide; we use it to find its tables and images and to build its ids.
            record_id = slide["slide_id"]
            # The text of the slide as parse_slides.py read it.
            original_text = slide.get("text", "")

            # The image texts of this slide (an empty list if it has none) and whether it has any.
            image_parts = image_text.get(record_id, [])
            has_image_text = bool(image_parts)

            # The text of the slide, then its tables row by row, then the text of its useful images, so the chunk
            # holds everything the slide shows.
            text = original_text
            if record_id in table_text:
                text += "\n\nTable, row by row:\n" + "\n".join(table_text[record_id])
            if image_parts:
                text += "\n\n" + "\n\n".join(image_parts)

            # A slide with no text, no table and no useful image gives no chunk.
            if not text.strip():
                continue

            # The position of the slide in the course: its sessions and its title.
            # A slide without sessions in the manifest is labelled "Slides".
            sessions = slide.get("sessions", [])
            session_label = (
                f"Session {', '.join(map(str, sessions))}"
                if sessions
                else "Slides"
            )

            # When the slide has a title (found by parse_slides.py), we add it after the sessions.
            title = slide.get("title", "").strip()
            outline_path = (
                f"{session_label} › {title}"
                if title
                else session_label
            )

            # The id of the parent is the slide id with "parent__" in front.
            parent_id = f"parent__{record_id}"

            # The whole slide (with its tables and image text), which we keep as the parent of its chunks.
            parents.append(
                Parent(
                    parent_id=parent_id,
                    outline_path=outline_path,
                    covers=[record_id],
                    text=text,
                )
            )

            # We cut each slide on its own, at sentence ends, so a chunk never mixes two slides.
            for i, piece in enumerate(split_into_chunks(text, chunk_size, overlap)):
                # We drop the pieces that are only a closing line, a page number or symbols (syllabus_text.py).
                if is_noise(piece, has_image_text):
                    continue
                chunks.append(
                    Chunk(
                        # The same fields as the syllabus chunks; here the chunk is cut by slide.
                        chunk_id=chunk_id("slide", record_id, i),
                        kb_sig=cfg.kb_sig,
                        strategy="slide",
                        doc_type="slide",
                        course_code=slide["course_code"],
                        # Every chunk of a slide has the sessions of its deck.
                        sessions=sessions,
                        source_id=slide["source_id"],
                        # The slide id, compared by the evaluation with the expected source of a question.
                        covers=[record_id],
                        parent_id=parent_id,
                        outline_path=outline_path,
                        text=piece,
                        embed_text=with_course(names.get(slide["course_code"], ""), outline_path, piece),
                        # Saved with the chunk in ChromaDB by build_indexes.py.
                        has_image_text=has_image_text,
                        # Filled later by dedup.py if this chunk is a near-duplicate of another one.
                        duplicate_of=None,
                    )
                )

    # Write both files, named with the kb_sig of this configuration.
    write_jsonl(chunks_path(cfg.kb_sig), chunks)
    write_jsonl(parents_path(cfg.kb_sig), parents)

    # A summary of the run, so we can check the counts and which image texts were included.
    print(f"Created {len(chunks)} chunks and {len(parents)} parents.")
    print(f"OCR enabled: {use_ocr}")
    print(f"Captions enabled: {use_captions}")
    print(f"Chunks: {chunks_path(cfg.kb_sig)}")
    print(f"Parents: {parents_path(cfg.kb_sig)}")

    # The number of chunks, printed by the build pipeline.
    return len(chunks)
