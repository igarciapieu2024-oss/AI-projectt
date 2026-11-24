# The source cards shown under each answer: the slide picture, the course, where the source comes from, its text
# and the text read from its image. The cited sources get a card; the others are listed in one line.
# Lets us write types like "list[Hit]" in any Python version.
from __future__ import annotations

# Reads the manifest and the image index.
import csv
# Escapes the source text, so it can't break our HTML.
import html as html_lib
# Finds the [S1], [S2]... citations and the image part of a slide.
import re
# Handles file paths the same way on Windows and Mac.
from pathlib import Path

# For st.cache_data, so each slide picture is made only once.
import streamlit as st

# Course names, and finds a PDF named in the manifest.
from app_components.engine import courses, pdf_path
# Our HTML helper, icons and image loader.
from app_components.theme import html, icon, image_data_uri
# Where the manifest, the PDFs and the image index are.
from src.config import PATHS
# The agreed format of one source.
from src.schemas import Hit

# The image part the chunking adds to a slide's text.
_IMAGE_TEXT = re.compile(r"\[Image: (\w+)\](.*)", re.S)
# The app's own slide pictures (generated, never uploaded).
APP_THUMBS = PATHS["interim"] / "app_thumbs"
# Readable names.
_DOC_TYPE_NAMES = {"syllabus_section": "Syllabus", "slide": "Slide", "course_row": "Course table"}


# 1. Small helpers

# The sources the answer actually cites, so they can be shown as cards.
def cited_labels(answer_text: str) -> set[str]:
    return set(re.findall(r"\[(S\d+)\]", answer_text))


# source_id -> the PDF file of that source, read from the manifest.
def _source_files() -> dict[str, str]:
    path = PATHS["manifest"]
    # No manifest yet: no slide pictures, the cards show the page icon.
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {row["source_id"]: row["file"] for row in csv.DictReader(f)}


# slide_id -> the thumbnail the image extraction already made of the whole slide (only for visual slides).
def _full_slide_thumbs() -> dict[str, Path]:
    # data/interim/images_index.csv, written once by the image extraction.
    path = PATHS["images_index"]
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        # thumb_path is relative to the repo folder.
        return {row["slide_id"]: PATHS["root"] / row["thumb_path"]
                for row in csv.DictReader(f) if row["kind"] == "full_slide"}


# A small picture of one slide, as an image the HTML can show. If the image extraction already rendered this slide
# whole, its thumbnail is reused; otherwise the page is rendered from the PDF and saved in data/interim/app_thumbs/,
# a folder only the app uses, so the files the image pipeline writes are never touched.
# Computed once per slide; the next time the same slide is cited it is instant.
@st.cache_data(show_spinner=False)
def _slide_thumbnail(slide_id: str) -> str | None:
    # Option 1: the extraction's own render.
    full = _full_slide_thumbs().get(slide_id)
    if full and full.exists():
        return image_data_uri(full)

    # Option 2: our own render, made the first time it's needed.
    cached = APP_THUMBS / f"{slide_id}.jpg"
    if not cached.exists():
        # "ai_ml_s04_p017" -> ("ai_ml_s04", "017").
        source_id, page = slide_id.rsplit("_p", 1)
        pdf_file = _source_files().get(source_id)
        # The slide isn't in the manifest (e.g. the fake engine's made-up slides).
        if pdf_file is None:
            return None
        # The PDF in data/raw/, found whatever the capitals in the manifest.
        path = pdf_path(pdf_file)
        # The PDF isn't on this laptop.
        if path is None:
            return None
        # Heavy library: imported here, only when a picture is really needed.
        import pymupdf

        with pymupdf.open(path) as pdf:
            # Page 17 is index 16; 0.6 = small picture.
            pixmap = pdf[int(page) - 1].get_pixmap(matrix=pymupdf.Matrix(0.6, 0.6))
        APP_THUMBS.mkdir(parents=True, exist_ok=True)
        # Saved, so the next start of the app doesn't render it again.
        pixmap.pil_save(cached, "JPEG", quality=80)
    return image_data_uri(cached)


# The picture on top of a source card: the slide itself for slides, nothing for syllabus sections.
def thumbnail(hit: Hit) -> str | None:
    # Syllabus sections have no picture.
    if hit.chunk.doc_type != "slide" or not hit.chunk.covers:
        return None
    try:
        # covers[0] is the slide_id the chunk comes from.
        return _slide_thumbnail(hit.chunk.covers[0])
    # A broken PDF page must never break the answer: the card just shows the page icon.
    except Exception:
        return None


# 2. One card

# The HTML of one cited source: thumbnail, tags, where it comes from, the text and the image text.
def _card(hit: Hit) -> str:
    thumb = thumbnail(hit)
    # The top of the card: the slide picture, or a page icon until the thumbnails exist (and for syllabi).
    top = f'<img src="{thumb}" alt="">' if thumb else icon("page", 30)

    # Is part of this source's text from an image?
    match = _IMAGE_TEXT.search(hit.chunk.text)
    # The text without the image part.
    main_text = hit.chunk.text[:match.start()].strip() if match else hit.chunk.text

    # S1 in yellow, like the chip in the answer.
    tags = [f'<span class="tag label on">{hit.label}</span>',
            # Syllabus / Slide.
            f'<span class="tag">{_DOC_TYPE_NAMES.get(hit.chunk.doc_type, hit.chunk.doc_type)}</span>']
    # e.g. "mgmt_control" -> "Management Control".
    course_name = courses().get(hit.chunk.course_code, hit.chunk.course_code)
    # The course of the source, so a source from the wrong course is visible at a glance.
    if course_name:
        tags.append(f'<span class="tag">{html_lib.escape(course_name)}</span>')
    # The answer can come from a diagram: we say so with its own tag.
    if match:
        tags.append(f'<span class="tag img">From a {html_lib.escape(match.group(1))}</span>')

    # The text read from the image, in its own box so the demo can point at it.
    image_box = ""
    if match:
        image_box = f'<div class="src-image">{html_lib.escape(match.group(2).strip()[:220])}</div>'

    return f"""
        <div class="src">
        <div class="src-thumb">{top}</div>
        <div class="src-body">
        <div class="src-tags">{''.join(tags)}</div>
        <div class="src-path">{html_lib.escape(hit.chunk.outline_path)}</div>
        <div class="src-text">{html_lib.escape(main_text[:400])}</div>
        {image_box}
        </div></div>
    """


# 3. All the cards of one answer

# The cited sources as cards; the ones retrieved but not cited in one quiet line underneath.
def render_sources(hits: list[Hit], answer_text: str) -> None:
    # Refusals and the LLM alone have no sources.
    if not hits:
        return
    cited = cited_labels(answer_text)
    # The answer cites them: big cards.
    used = [h for h in hits if h.label in cited]
    # Found by the search but not needed: one line.
    unused = [h for h in hits if h.label not in cited]
    also = ""
    if unused:
        also = '<div class="also">Also retrieved, not cited: ' + "".join(
            f'<span class="tag label">{h.label}</span>{html_lib.escape(h.chunk.outline_path)}&emsp;' for h in unused) + "</div>"
    html(f"""
        <div class="src-head">Sources</div>
        <div class="sources">{''.join(_card(h) for h in used)}</div>
        {also}
    """)