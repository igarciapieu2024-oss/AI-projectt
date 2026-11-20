# Remove footer lines that repeat across the slides of a course.
# Lines such as the lecturer's name or a copyright notice appear at the bottom of many
# slides; if they stayed in the text they would match many questions by accident.
# remove_repeated_footers() is called by parse_slides.py before the slides are saved.
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import replace

from src.schemas import SlideRecord


# A line is a footer if it appears at the bottom of more than 15% of a course's slides.
FOOTER_SHARE = 0.15


# Normalize a line so small differences do not hide a repetition: lower case,
# punctuation replaced by spaces and repeated spaces collapsed.
# "\w" keeps the underscore, so it is replaced by a space at the end.
def _normalize_line(line: str) -> str:
    words = re.sub(r"[^\w]+", " ", line.casefold(), flags=re.UNICODE)
    return " ".join(words.split()).replace("_", " ")


# Return the possible footer lines of one slide: the last three non-empty lines,
# normalized, since footers are always at the bottom of the page.
def _footer_candidates(lines: list[str]) -> set[str]:
    positions = [index for index, line in enumerate(lines) if line.strip()]
    candidates = set()
    for index in positions[-3:]:
        normalized = _normalize_line(lines[index])
        # Ignore very short lines and lines without letters, such as page numbers.
        if len(normalized) >= 4 and any(char.isalpha() for char in normalized):
            candidates.add(normalized)
    return candidates


# Return the footer lines of a group of slides: the candidates that appear on
# more than min_share of the slides. Each slide counts a line at most once,
# because the candidates are a set.
def find_boilerplate(pages: list[list[str]], min_share: float = FOOTER_SHARE) -> set[str]:
    if not 0 <= min_share <= 1:
        raise ValueError("min_share must be between zero and one")
    if not pages:
        return set()

    # Count on how many slides each candidate line appears.
    counts: Counter[str] = Counter()
    for lines in pages:
        counts.update(_footer_candidates(lines))

    return {
        line for line, count in counts.items()
        if count / len(pages) > min_share
    }


# Remove the repeated footer lines from every slide and return the cleaned slides.
# Footers are found course by course, because each course has its own lecturer and
# template. A footer line is only removed from the last three non-empty lines of a
# slide, so the same words in the body of a slide are kept. Only the text changes:
# ids and the other fields stay the same.
def remove_repeated_footers(
    slides: list[SlideRecord], min_share: float = FOOTER_SHARE
) -> list[SlideRecord]:
    # Group the slides by course.
    by_course: dict[str, list[SlideRecord]] = defaultdict(list)
    for slide in slides:
        by_course[slide.course_code].append(slide)

    # Find the footer lines of each course.
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

    # Remove those lines from the last three non-empty positions of each slide.
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
        # replace() returns a copy with the new text; unchanged slides are kept as they are.
        cleaned.append(replace(slide, text=new_text) if new_text != slide.text else slide)

    return cleaned

# Placeholder that was never implemented. The agenda, low-text and garbled flags
# are computed in parse_slides.py instead.
def flag_slide(title: str, text: str) -> dict:
    raise NotImplementedError("flag_slide: not built yet")
