"""Slide images out of the PDFs + full renders of visual slides + thumbnails -> data/interim/images/, thumbs/, images_index.csv.

Owner: Santiago.
"""
from __future__ import annotations


def extract_images(cfg) -> int:
    """Extract images, drop tiny and repeated ones, render flagged slides; return the image count."""
    raise NotImplementedError("extract_images: not built yet")
