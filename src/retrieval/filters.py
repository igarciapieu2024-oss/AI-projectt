# Find which courses (and sessions) a question names. mentioned_course_codes() is used by
# src/generation/answer.py for the course filter and the per-course answers, and by
# src/generation/table_qa.py to keep only the courses a cross-course question names.
from __future__ import annotations

import re
import unicodedata

from src.schemas import Filters


# British and American spellings of words in the course names, counted as the same word, so
# "Organizational Behavior" finds the course stored as "Organizational Behaviour".
_SPELLING = {"behaviour": "behavior", "organisational": "organizational"}


# Normalise a text so names can be compared: lower case, accents removed, only letters and digits,
# and the spellings above unified.
def _normalized(value: str) -> str:
    # NFKD splits an accented letter into the letter and its accent, so the accent can be dropped.
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    words = re.findall(r"[a-z0-9]+", ascii_text)
    return " ".join(_SPELLING.get(word, word) for word in words)


# Build Filters from a question: a course only when exactly one is named (with two or more the course is
# left open), and the session numbers written after "session" or "sessions". Not used by the final answer
# pipeline, which calls mentioned_course_codes() directly; it is covered by the tests.
def extract_filters(question: str, courses: list[dict]) -> Filters:
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


# Return the codes of every course whose code or full name appears in the question, in the order of the
# courses list. courses are rows of the course table (course_code and course_name).
def mentioned_course_codes(question: str, courses: list[dict]) -> list[str]:
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    # Spaces around the text, so a name only matches as whole words and not inside a longer word.
    normalized_question = f" {_normalized(question)} "
    course_codes = []
    for course in courses:
        code = course.get("course_code")
        name = course.get("course_name")
        # A row without a code cannot be matched.
        if not isinstance(code, str) or not code.strip():
            continue
        # A course is named if its code or its full name appears.
        aliases = [code]
        if isinstance(name, str) and name.strip():
            aliases.append(name)
        if any(f" {_normalized(alias)} " in normalized_question for alias in aliases):
            course_codes.append(code)
    return course_codes
