
"""Automated tests for Elena's find_ids search tool."""

import json

from eval.find_ids import find_ids


def test_find_syllabus_section(tmp_path):
    """Check that syllabus sections can be found."""
    data = {
        "section_id": "c01_syl_sec04",
        "course_code": "c01",
        "section_name": "EVALUATION",
        "text": "The final exam represents 40% of the course grade.",
    }

    file = tmp_path / "syllabus_sections.jsonl"
    file.write_text(json.dumps(data) + "\n", encoding="utf-8")

    results = find_ids("final exam", tmp_path)

    assert len(results) == 1
    assert results[0]["id"] == "c01_syl_sec04"
    assert results[0]["type"] == "section"


def test_find_slide(tmp_path):
    """Check that slides can be found."""
    data = {
        "slide_id": "c01_s04_p017",
        "course_code": "c01",
        "title": "Supply and Demand",
        "text": "The equilibrium price is determined by supply and demand.",
    }

    file = tmp_path / "slides.jsonl"
    file.write_text(json.dumps(data) + "\n", encoding="utf-8")

    results = find_ids("equilibrium price", tmp_path)

    assert len(results) == 1
    assert results[0]["id"] == "c01_s04_p017"
    assert results[0]["type"] == "slide"


def test_find_image(tmp_path):
    """Check that image descriptions return the parent slide ID."""
    data = {
        "image_id": "c01_s04_p017_img01",
        "slide_id": "c01_s04_p017",
        "source_id": "c01_s04",
        "ocr_text": "Input Conv Pool Dense",
        "description": "A neural network architecture diagram.",
        "image_class": "diagram",
    }

    file = tmp_path / "image_texts.jsonl"
    file.write_text(json.dumps(data) + "\n", encoding="utf-8")

    results = find_ids("neural network", tmp_path)

    assert len(results) == 1
    assert results[0]["id"] == "c01_s04_p017"
    assert results[0]["type"] == "image"


def test_no_matches(tmp_path):
    """Check that unrelated queries return no results."""
    data = {
        "slide_id": "c01_s05_p008",
        "course_code": "c01",
        "title": "Inflation",
        "text": "Inflation measures changes in the price level.",
    }

    file = tmp_path / "slides.jsonl"
    file.write_text(json.dumps(data) + "\n", encoding="utf-8")

    results = find_ids("quantum spaceship", tmp_path)

    assert results == []
