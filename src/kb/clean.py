"""Remove repeated footers and page numbers; flag agenda, low-text and garbled slides.

Owner: Santiago.
"""
from __future__ import annotations


def find_boilerplate(pages: list[list[str]], min_share: float = 0.5) -> set[str]:
    """Lines present on more than min_share of a deck's pages."""
    raise NotImplementedError("find_boilerplate: not built yet")

def flag_slide(title: str, text: str) -> dict:
    """Return {'is_agenda': bool, 'is_low_text': bool, 'is_garbled': bool}."""
    raise NotImplementedError("flag_slide: not built yet")
