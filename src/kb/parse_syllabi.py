"""Syllabus PDFs -> data/interim/syllabus_sections.jsonl (one SyllabusSection per section, tables as markdown).

Owner: Santiago.
"""
from __future__ import annotations


def parse_syllabi(cfg) -> int:
    """Parse every syllabus in the manifest; write SyllabusSection rows; return how many."""
    raise NotImplementedError("parse_syllabi: not built yet")
