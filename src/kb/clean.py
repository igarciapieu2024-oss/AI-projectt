"""Remove repeated slide footers and flag agenda, low-text and garbled slides.

Owner: Santiago.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import replace

from src.schemas import SlideRecord


FOOTER_SHARE = 0.15


def _normalize_line(line: str) -> str:
    """Normalize case, whitespace and punctuation for footer comparison."""
    words = re.sub(r"[^\w]+", " ", line.casefold(), flags=re.UNICODE)
    return " ".join(words.split()).replace("_", " ")


def _footer_candidates(lines: list[str]) -> set[str]:
    """Return nonempty textual lines in the last three nonempty line positions."""
    positions = [index for index, line in enumerate(lines) if line.strip()]
    candidates = set()
    for index in positions[-3:]:
        normalized = _normalize_line(lines[index])
        # Ignore blank/punctuation-only and numeric-only lines (such as page numbers).
        if len(normalized) >= 4 and any(char.isalpha() for char in normalized):
            candidates.add(normalized)
    return candidates


def find_boilerplate(pages: list[list[str]], min_share: float = FOOTER_SHARE) -> set[str]:
    """Return normalized footer lines present on strictly more than ``min_share`` of pages.

    Only the final three nonempty lines are considered footer candidates. Each
    page contributes a line at most once, even if the text repeats within it.
    """
    if not 0 <= min_share <= 1:
        raise ValueError("min_share must be between zero and one")
    if not pages:
        return set()

    counts: Counter[str] = Counter()
    for lines in pages:
        counts.update(_footer_candidates(lines))

    return {
        line for line, count in counts.items()
        if count / len(pages) > min_share
    }


def remove_repeated_footers(
    slides: list[SlideRecord], min_share: float = FOOTER_SHARE
) -> list[SlideRecord]:
    """Remove repeated footer lines independently within each course.

    A matching line is removed only in the final three nonempty line positions.
    Slide records are copied with ``text`` changed, leaving all IDs and other
    metadata untouched. Slide titles are protected from removal.
    """
    by_course: dict[str, list[SlideRecord]] = defaultdict(list)
    for slide in slides:
        by_course[slide.course_code].append(slide)

    removed_by_course: dict[str, set[str]] = {}
    for course_code, course_slides in by_course.items():
        pages = [slide.text.splitlines() for slide in course_slides]
        footers = find_boilerplate(pages, min_share)

        # A line that is used as a slide title is meaningful content, even if it
        # also occurs near the bottom of enough slides to cross the threshold.
        title_lines = {
            _normalize_line(slide.title)
            for slide in course_slides
            if slide.title.strip()
        }
        removed_by_course[course_code] = footers - title_lines

    cleaned = []
    for slide in slides:
        footer_lines = removed_by_course[slide.course_code]
        lines = slide.text.splitlines()
        footer_positions = set(
            index for index, line in enumerate(lines) if line.strip()
        )
        footer_positions = set(sorted(footer_positions)[-3:])
        updated_lines = [
            line for index, line in enumerate(lines)
            if not (
                index in footer_positions
                and _normalize_line(line) in footer_lines
            )
        ]
        new_text = "\n".join(updated_lines)
        cleaned.append(replace(slide, text=new_text) if new_text != slide.text else slide)

    return cleaned

def flag_slide(title: str, text: str) -> dict:
    """Return {'is_agenda': bool, 'is_low_text': bool, 'is_garbled': bool}."""
    raise NotImplementedError("flag_slide: not built yet")
