"""The brand look: loads the CSS and draws HTML blocks (owner: Irene)."""
from __future__ import annotations  # lets us write types like "str | None" in any Python version

import base64  # turns small images into text so they can go inside the HTML
from pathlib import Path  # handles file paths the same way on Windows and Mac

import streamlit as st  # the library that builds the web app

ROOT = Path(__file__).resolve().parent.parent  # the repo folder (this file is in app_components/)
CSS_FILE = ROOT / "assets" / "style.css"  # every style rule of the app


#1 the CSS and the HTML helper

def apply_theme() -> None:
    """Adds our CSS to the page; app.py calls it once, before any page is drawn."""
    css = CSS_FILE.read_text(encoding="utf-8")  # reads the style file
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)  # inserts it into the page


def html(markup: str) -> None:
    """Draws a block of our own HTML.

    Streamlit reads the text as Markdown first: an indented line becomes a code block and an
    empty line ends the HTML. So we strip the indentation and the empty lines before drawing it.
    """
    lines = [line.strip() for line in markup.splitlines()]  # removes the indentation of every line
    st.markdown("".join(line for line in lines if line), unsafe_allow_html=True)  # joins the non-empty lines


def image_data_uri(path: str | Path) -> str | None:
    """A small image as text ("data:image/png;base64,..."), so an <img> works without a web server."""
    path = Path(path)
    if not path.is_absolute():  # paths in app_content.py are relative to the repo folder
        path = ROOT / path
    if not path.exists():  # the image hasn't been made yet
        return None
    kind = "svg+xml" if path.suffix == ".svg" else path.suffix.lstrip(".").replace("jpg", "jpeg")  # the image type
    return f"data:image/{kind};base64,{base64.b64encode(path.read_bytes()).decode()}"  # the file as text


#2 small line icons drawn in SVG (offline: no icon library from the internet)

_ICON_PATHS = {
    # a quote mark inside a speech bubble: "cites every answer"
    "cite": '<path d="M4 5h16v11H9l-5 4z"/><path d="M9 9.5h2.5V12M13 9.5h2.5V12"/>',
    # a laptop with no signal waves: "works offline"
    "offline": '<rect x="4" y="5" width="16" height="10" rx="1.5"/><path d="M2 19h20"/><path d="M3 3l18 18"/>',
    # a picture with mountains and a sun: "reads images"
    "image": '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="1.6"/><path d="M21 16l-5-5-9 9"/>',
    # a shield: privacy
    "shield": '<path d="M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6z"/>',
    # three stacked sheets: "built from your courses"
    "layers": '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/><path d="M3 17.5l9 5 9-5" opacity=".55"/>',
    # a presentation board: lecture slides
    "slides": '<rect x="3" y="4" width="18" height="12" rx="1.5"/><path d="M12 16v4M8 20h8M7 12l3-3 2 2 4-4"/>',
    # a tick: "checked"
    "check": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    # a page with lines: a syllabus or a slide
    "page": '<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4M9 12h6M9 16h4"/>',
}


def icon(name: str, size: int = 28, color: str = "currentColor") -> str:
    """One of the icons above as SVG text, ready to put inside an HTML block."""
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            f'{_ICON_PATHS[name]}</svg>')
