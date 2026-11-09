
"""Find syllabus, slide and image source IDs matching search words.

Usage:
    python -m eval.find_ids "final exam"

Owner: Elena.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import PATHS


def find_ids(query: str, data_dir: Path | None = None) -> list[dict]:
    """Search syllabus sections, slides and image text for query words."""
    words = query.casefold().split()
    if not words:
        return []

    if data_dir is None:
        files = [
            (PATHS["syllabus_sections"], "section"),
            (PATHS["slides"], "slide"),
            (PATHS["image_texts"], "image"),
        ]
    else:
        data_dir = Path(data_dir)
        files = [
            (data_dir / "syllabus_sections.jsonl", "section"),
            (data_dir / "slides.jsonl", "slide"),
            (data_dir / "image_texts.jsonl", "image"),
        ]

    matches = []

    for path, record_type in files:
        if not path.exists():
            print(f"File not found: {path}")
            continue

        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)

                if record_type == "section":
                    source_id = record.get("section_id", "")
                    course = record.get("course_code", "")
                    title = record.get("section_name", "")
                    text = record.get("text", "")
                elif record_type == "slide":
                    source_id = record.get("slide_id", "")
                    course = record.get("course_code", "")
                    title = record.get("title", "")
                    text = record.get("text", "")
                else:
                    source_id = record.get("slide_id", "")
                    course = record.get("course_code", "")
                    title = record.get("image_class", "Image")
                    text = (
                        record.get("ocr_text", "")
                        + " "
                        + record.get("description", "")
                    )

                searchable = f"{title} {text}".casefold()

                if all(word in searchable for word in words):
                    matches.append({
                        "id": source_id,
                        "course": course,
                        "title": title,
                        "snippet": " ".join(text.split())[:160],
                        "type": record_type,
                    })

    return matches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find syllabus section and slide IDs."
    )
    parser.add_argument("query", help="Words to search for")
    args = parser.parse_args()

    results = find_ids(args.query)

    if not results:
        print("No matching sections, slides or images found.")
        return

    for result in results:
        print(f"\nType: {result['type']}")
        print(f"ID: {result['id']}")
        print(f"Course: {result['course']}")
        print(f"Title/Section: {result['title']}")
        print(f"Text: {result['snippet']}")

    print(f"\nTotal matches: {len(results)}")


if __name__ == "__main__":
    main()
