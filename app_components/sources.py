"""The source cards shown under each answer (owner: Irene)."""
from __future__ import annotations  # lets us write types like "list[Hit]" in any Python version

import csv  # reads the manifest and the image index
import html as html_lib  # escapes the source text, so it can't break our HTML
import re  # finds the [S1], [S2]... citations and the image part of a slide
from pathlib import Path  # handles file paths the same way on Windows and Mac

import streamlit as st  # for st.cache_data, so each slide picture is made only once

from app_components.engine import pdf_path  # finds a PDF named in the manifest
from app_components.theme import html, icon, image_data_uri  # our HTML helper, icons and image loader
from src.config import PATHS  # where the manifest, the PDFs and the image index are
from src.schemas import Hit  # the agreed format of one source

_IMAGE_TEXT = re.compile(r"\[Image: (\w+)\](.*)", re.S)  # the image part the chunking adds to a slide's text
APP_THUMBS = PATHS["interim"] / "app_thumbs"  # the app's own slide pictures (generated, never uploaded)
_DOC_TYPE_NAMES = {"syllabus_section": "Syllabus", "slide": "Slide", "course_row": "Course table"}  # readable names


#1 small helpers

def cited_labels(answer_text: str) -> set[str]:
    """The sources the answer actually cites, e.g. {"S1", "S3"}, so we can highlight them."""
    return set(re.findall(r"\[(S\d+)\]", answer_text))


def _source_files() -> dict[str, str]:
    """source_id -> the PDF file of that source, read from the manifest (e.g. "ai_ml_s04" -> "AI-.../Session 4.pdf")."""
    path = PATHS["manifest"]
    if not path.exists():  # no manifest yet: no slide pictures, the cards show the page icon
        return {}
    with path.open(encoding="utf-8") as f:
        return {row["source_id"]: row["file"] for row in csv.DictReader(f)}


def _full_slide_thumbs() -> dict[str, Path]:
    """slide_id -> the thumbnail the image extraction already made of the whole slide (only for visual slides)."""
    path = PATHS["images_index"]  # data/interim/images_index.csv, written once by the image extraction
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {row["slide_id"]: PATHS["root"] / row["thumb_path"]  # thumb_path is relative to the repo folder
                for row in csv.DictReader(f) if row["kind"] == "full_slide"}


@st.cache_data(show_spinner=False)  # computed once per slide; the next time the same slide is cited it is instant
def _slide_thumbnail(slide_id: str) -> str | None:
    """A small picture of one slide, as an image the HTML can show.

    1. If the image extraction already rendered this slide whole, we reuse its thumbnail.
    2. Otherwise we render the page from the PDF ourselves and save it in data/interim/app_thumbs/,
       a folder only the app uses, so we never touch the files the image pipeline writes.
    """
    full = _full_slide_thumbs().get(slide_id)  # option 1: the extraction's own render
    if full and full.exists():
        return image_data_uri(full)

    cached = APP_THUMBS / f"{slide_id}.jpg"  # option 2: our own render, made the first time it's needed
    if not cached.exists():
        source_id, page = slide_id.rsplit("_p", 1)  # "ai_ml_s04_p017" -> ("ai_ml_s04", "017")
        pdf_file = _source_files().get(source_id)
        if pdf_file is None:  # the slide isn't in the manifest (e.g. the fake engine's made-up slides)
            return None
        path = pdf_path(pdf_file)  # the PDF in data/raw/, found whatever the capitals in the manifest
        if path is None:  # the PDF isn't on this laptop
            return None
        import pymupdf  # heavy library: imported here, only when a picture is really needed

        with pymupdf.open(path) as pdf:
            pixmap = pdf[int(page) - 1].get_pixmap(matrix=pymupdf.Matrix(0.6, 0.6))  # page 17 is index 16; 0.6 = small picture
        APP_THUMBS.mkdir(parents=True, exist_ok=True)
        pixmap.pil_save(cached, "JPEG", quality=80)  # saved, so the next start of the app doesn't render it again
    return image_data_uri(cached)


def thumbnail(hit: Hit) -> str | None:
    """The picture on top of a source card: the slide itself for slides, nothing for syllabus sections."""
    if hit.chunk.doc_type != "slide" or not hit.chunk.covers:  # syllabus sections have no picture
        return None
    try:
        return _slide_thumbnail(hit.chunk.covers[0])  # covers[0] is the slide_id the chunk comes from
    except Exception:  # a broken PDF page must never break the answer: the card just shows the page icon
        return None


#2 one card

def _card(hit: Hit) -> str:
    """The HTML of one cited source: thumbnail, tags, where it comes from, the text and the image text."""
    thumb = thumbnail(hit)
    # the top of the card: the slide picture, or a page icon until the thumbnails exist (and for syllabi)
    top = f'<img src="{thumb}" alt="">' if thumb else icon("page", 30)

    match = _IMAGE_TEXT.search(hit.chunk.text)  # is part of this source's text from an image?
    main_text = hit.chunk.text[:match.start()].strip() if match else hit.chunk.text  # the text without the image part

    tags = [f'<span class="tag label on">{hit.label}</span>',  # S1 in yellow, like the chip in the answer
            f'<span class="tag">{_DOC_TYPE_NAMES.get(hit.chunk.doc_type, hit.chunk.doc_type)}</span>']  # Syllabus / Slide
    if match:  # the answer can come from a diagram: we say so with its own tag
        tags.append(f'<span class="tag img">From a {html_lib.escape(match.group(1))}</span>')

    image_box = ""  # the text read from the image, in its own box so the demo can point at it
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


#3 all the cards of one answer

def render_sources(hits: list[Hit], answer_text: str) -> None:
    """The cited sources as cards; the ones retrieved but not cited in one quiet line underneath."""
    if not hits:  # refusals and the LLM alone have no sources
        return
    cited = cited_labels(answer_text)
    used = [h for h in hits if h.label in cited]  # the answer cites them: big cards
    unused = [h for h in hits if h.label not in cited]  # found by the search but not needed: one line
    also = ""
    if unused:
        also = '<div class="also">Also retrieved, not cited: ' + "".join(
            f'<span class="tag label">{h.label}</span>{html_lib.escape(h.chunk.outline_path)}&emsp;' for h in unused) + "</div>"
    html(f"""
        <div class="src-head">Sources</div>
        <div class="sources">{''.join(_card(h) for h in used)}</div>
        {also}
    """)
