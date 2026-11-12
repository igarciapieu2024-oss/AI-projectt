"""Course and session filters detected in the question.

Owner: Sergio.
"""
from __future__ import annotations

import re
import unicodedata

from src.schemas import Filters


def _normalized(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text))


def extract_filters(question: str, courses: list[dict]) -> Filters:
    """Extract an unambiguous course name/code and explicitly mentioned sessions."""
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    course_codes = mentioned_course_codes(question, courses)
    sessions = sorted({
        int(match.group(1))
        for match in re.finditer(r"\b(?:sessions?|sesiones?)\s*(\d+)\b", question, re.IGNORECASE)
    })
    return Filters(
        course_code=course_codes[0] if len(course_codes) == 1 else None,
        sessions=sessions,
    )


def mentioned_course_codes(question: str, courses: list[dict]) -> list[str]:
    """Return all course codes whose names or codes are explicitly present."""
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    normalized_question = f" {_normalized(question)} "
    course_codes = []
    for course in courses:
        code = course.get("course_code")
        name = course.get("course_name")
        if not isinstance(code, str) or not code.strip():
            continue
        aliases = [code]
        if isinstance(name, str) and name.strip():
            aliases.append(name)
        if any(f" {_normalized(alias)} " in normalized_question for alias in aliases):
            course_codes.append(code)
    return course_codes
