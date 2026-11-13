from dataclasses import asdict

from src.kb import clean, parse_slides
from src.schemas import SlideRecord


def _slide(index, course_code="course_a", text=None, title=None):
    source_id = f"{course_code}_s{index + 1:02d}"
    return SlideRecord(
        slide_id=f"{source_id}_p001",
        source_id=source_id,
        course_code=course_code,
        sessions=[index + 1],
        page=1,
        title=title or f"Meaningful title {index}",
        outline=[f"Topic {index}"],
        text=text if text is not None else f"Meaningful content for slide {index}.",
        is_agenda=False,
        is_low_text=False,
        is_garbled=False,
        n_images=index % 2,
    )


def _slides_with_footer(count, footer, indices, course_code="course_a"):
    return [
        _slide(
            i,
            course_code=course_code,
            text=f"Meaningful content for slide {i}.\n{footer if i in indices else ''}",
        )
        for i in range(count)
    ]


def test_footer_at_exactly_fifteen_percent_remains():
    slides = _slides_with_footer(20, "Course footer © 2025", {0, 1, 2})

    cleaned = clean.remove_repeated_footers(slides)

    assert all("Course footer © 2025" in slide.text for slide in cleaned[:3])


def test_footer_above_fifteen_percent_is_removed():
    slides = _slides_with_footer(20, "Course footer © 2025", {0, 1, 2, 3})

    cleaned = clean.remove_repeated_footers(slides)

    assert all("Course footer © 2025" not in slide.text for slide in cleaned[:4])


def test_repeated_footer_in_third_from_bottom_position_is_removed():
    slides = [
        _slide(
            i,
            text=(
                f"Meaningful content for slide {i}.\n"
                + ("Repeated deck footer © 2025\n" if i < 4 else "")
                + f"Slide number {i + 1}\nDeck label {i}"
            ),
        )
        for i in range(20)
    ]

    cleaned = clean.remove_repeated_footers(slides)

    for slide in cleaned[:4]:
        assert "Repeated deck footer © 2025" not in slide.text
        assert "Slide number" in slide.text
        assert f"Deck label {slide.sessions[0] - 1}" in slide.text
    assert "Meaningful content for slide 0." in cleaned[0].text


def test_repeated_occurrences_within_one_slide_count_once():
    slides = [_slide(i) for i in range(20)]
    slides[0].text = "Content one.\nFooter © 2025\nFooter © 2025\nFooter © 2025\nFooter © 2025"
    slides[1].text = "Content two.\nFooter © 2025"

    cleaned = clean.remove_repeated_footers(slides)

    assert "Footer © 2025" in cleaned[0].text
    assert "Footer © 2025" in cleaned[1].text


def test_empty_and_numeric_only_lines_are_not_footer_candidates():
    pages = [["", "2025"], ["   ", "2025"], ["!!!", ""]]

    assert clean.find_boilerplate(pages) == set()


def test_punctuation_case_and_whitespace_variants_match():
    variants = {
        0: "ACME, Inc. — All Rights Reserved © 2025",
        1: " acme inc all rights reserved 2025 ",
        2: "ACME INC., ALL RIGHTS RESERVED © 2025!",
        3: "Acme, Inc.  All Rights Reserved, 2025",
    }
    slides = [
        _slide(i, text=f"Distinct content {i}.\n{variants.get(i, '')}")
        for i in range(20)
    ]

    cleaned = clean.remove_repeated_footers(slides)

    assert all(variants[i].strip() not in cleaned[i].text for i in variants)
    assert all(f"Distinct content {i}." in cleaned[i].text for i in variants)


def test_footer_frequency_is_calculated_independently_per_course():
    course_a = _slides_with_footer(10, "Shared footer © 2025", {0, 1}, "course_a")
    course_b = _slides_with_footer(10, "Shared footer © 2025", {0}, "course_b")

    cleaned = clean.remove_repeated_footers(course_a + course_b)

    assert all("Shared footer" not in slide.text for slide in cleaned[:2])
    assert "Shared footer © 2025" in cleaned[10].text


def test_meaningful_content_and_repeated_slide_titles_are_preserved():
    slides = [
        _slide(i, text=f"Conclusion\nUnique useful content {i}.", title="Conclusion")
        for i in range(20)
    ]

    cleaned = clean.remove_repeated_footers(slides)

    assert all(slide.text == slides[i].text for i, slide in enumerate(cleaned))


def test_slide_ids_and_metadata_are_unchanged():
    slides = _slides_with_footer(20, "Footer © 2025", {0, 1, 2, 3})

    cleaned = clean.remove_repeated_footers(slides)

    assert [slide.slide_id for slide in cleaned] == [slide.slide_id for slide in slides]
    for before, after in zip(slides, cleaned):
        before_data = asdict(before)
        after_data = asdict(after)
        before_data.pop("text")
        after_data.pop("text")
        assert after_data == before_data


def test_parser_cleans_after_collecting_slides_from_all_decks(tmp_path, monkeypatch):
    rows = [
        {"doc_type": "slides", "file": f"deck_{i}.pdf", "source_id": f"course_a_s{i + 1:02d}",
         "course_code": "course_a", "sessions": str(i + 1)}
        for i in range(20)
    ]
    monkeypatch.setitem(parse_slides.PATHS, "manifest", tmp_path / "manifest.csv")
    monkeypatch.setitem(parse_slides.PATHS, "root", tmp_path)
    monkeypatch.setitem(parse_slides.PATHS, "slides", tmp_path / "slides.jsonl")
    monkeypatch.setattr(parse_slides, "read_csv", lambda path: rows)

    def parsed_deck(pdf_path, source_id, course_code, sessions):
        index = int(source_id.rsplit("s", 1)[1]) - 1
        footer = "Deck footer © 2025" if index < 4 else ""
        return [_slide(index, course_code, f"Deck content {index}.\n{footer}")]

    monkeypatch.setattr(parse_slides, "_parse_slide_pdf", parsed_deck)
    written = {}
    monkeypatch.setattr(
        parse_slides,
        "write_jsonl",
        lambda path, records: written.update(path=path, records=records) or len(records),
    )

    result = parse_slides.parse_slides(None)

    assert result == 20
    assert all("Deck footer" not in slide.text for slide in written["records"][:4])
    assert all(slide.course_code == "course_a" for slide in written["records"])
