"""Slide PDFs -> data/interim/slides.jsonl (one SlideRecord per page: title, outline, sessions, flags).

Owner: Santiago.
"""
from __future__ import annotations


def parse_slides(cfg) -> int:
    """Parse every slide deck in the manifest; write SlideRecord rows; return how many."""
    raise NotImplementedError("parse_slides: not built yet")
