"""Runs the image steps once and writes data/image_texts.jsonl (resumable). Run: python -m src.vision.run_vision

Owner: Santiago (extraction, OCR) and Irene (classification, descriptions).
"""
from __future__ import annotations


def run_vision(cfg) -> int:
    """Write one ImageText per image, skipping IDs already in the file; return how many were added."""
    raise NotImplementedError("run_vision: not built yet")
