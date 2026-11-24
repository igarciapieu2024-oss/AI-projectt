# The brand look: loads the CSS (assets/style.css) and draws our HTML blocks and icons; every page uses it.
# Lets us write types like "str | None" in any Python version.
from __future__ import annotations

# Turns small images into text so they can go inside the HTML.
import base64
# Handles file paths the same way on Windows and Mac.
from pathlib import Path

# The library that builds the web app.
import streamlit as st

# The repo folder (this file is in app_components/).
ROOT = Path(__file__).resolve().parent.parent
# Every style rule of the app.
CSS_FILE = ROOT / "assets" / "style.css"


# 1. The CSS and the HTML helper

# Adds our CSS to the page; app.py calls it once, before any page is drawn.
def apply_theme() -> None:
    # Reads the style file.
    css = CSS_FILE.read_text(encoding="utf-8")
    # Inserts it into the page.
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# Draws a block of our own HTML. Streamlit reads the text as Markdown first: an indented line becomes a code block
# and an empty line ends the HTML, so the indentation and the empty lines are removed before drawing it.
def html(markup: str) -> None:
    # Removes the indentation of every line.
    lines = [line.strip() for line in markup.splitlines()]
    # Joins the non-empty lines.
    st.markdown("".join(line for line in lines if line), unsafe_allow_html=True)


# A small image as text ("data:image/png;base64,..."), so an <img> works without a web server; None if it doesn't exist.
def image_data_uri(path: str | Path) -> str | None:
    path = Path(path)
    # Paths in app_content.py are relative to the repo folder.
    if not path.is_absolute():
        path = ROOT / path
    # The image hasn't been made yet.
    if not path.exists():
        return None
    # The image type.
    kind = "svg+xml" if path.suffix == ".svg" else path.suffix.lstrip(".").replace("jpg", "jpeg")
    # The file as text.
    return f"data:image/{kind};base64,{base64.b64encode(path.read_bytes()).decode()}"


# 2. Small line icons drawn in SVG (offline: no icon library from the internet)

_ICON_PATHS = {
    # A quote mark inside a speech bubble: "cites every answer".
    "cite": '<path d="M4 5h16v11H9l-5 4z"/><path d="M9 9.5h2.5V12M13 9.5h2.5V12"/>',
    # A laptop with no signal waves: "works offline".
    "offline": '<rect x="4" y="5" width="16" height="10" rx="1.5"/><path d="M2 19h20"/><path d="M3 3l18 18"/>',
    # A picture with mountains and a sun: "reads images".
    "image": '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="1.6"/><path d="M21 16l-5-5-9 9"/>',
    # A shield: privacy.
    "shield": '<path d="M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6z"/>',
    # Three stacked sheets: "built from your courses".
    "layers": '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/><path d="M3 17.5l9 5 9-5" opacity=".55"/>',
    # A presentation board: lecture slides.
    "slides": '<rect x="3" y="4" width="18" height="12" rx="1.5"/><path d="M12 16v4M8 20h8M7 12l3-3 2 2 4-4"/>',
    # A tick: "checked".
    "check": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    # A page with lines: a syllabus or a slide.
    "page": '<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4M9 12h6M9 16h4"/>',
    # A building with columns: the university.
    "campus": '<path d="M3 9l9-5 9 5"/><path d="M5 9v9M9.5 9v9M14.5 9v9M19 9v9"/><path d="M3 20h18"/>',
    # Two people: the users.
    "users": '<circle cx="9" cy="8" r="3"/><path d="M3.5 19c.6-3.2 2.8-5 5.5-5s4.9 1.8 5.5 5"/><circle cx="17" cy="9" r="2.3"/><path d="M15.5 14.2c2.4.2 4.2 1.9 4.8 4.8"/>',
}


# One of the icons above as SVG text, ready to put inside an HTML block.
def icon(name: str, size: int = 28, color: str = "currentColor") -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            f'{_ICON_PATHS[name]}</svg>')
