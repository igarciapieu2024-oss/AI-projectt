
"""Find syllabus section and slide IDs matching a search query.

Usage:
    python -m eval.find_ids "final exam"

Owner: Elena.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATA_DIR = Path("data/interim")


def find_ids(query: str, data_dir: Path = DATA_DIR) -> list[dict]:
    """Return sections and slides containing all query words."""
    words = query.casefold().split()
    if not words:
        return []

    matches = []

    for filename, id_field, title_field in [
        ("syllabus_sections.jsonl", "section_id", "section_name"),
        ("slides.jsonl", "slide_id", "title"),
    ]:
        path = data_dir / filename

        if not path.exists():
            print(f"File not found: {path}")
            continue

        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)
                text = record.get("text", "")
                title = record.get(title_field, "")

                searchable = f"{title} {text}".casefold()

                if all(word in searchable for word in words):
                    matches.append({
                        "id": record.get(id_field, ""),
                        "course": record.get("course_code", ""),
                        "title": title,
                        "snippet": " ".join(text.split())[:160],
                    })

    return matches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find IDs of syllabus sections and slides."
    )
    parser.add_argument("query", help="Words to search for")
    args = parser.parse_args()

    results = find_ids(args.query)

    if not results:
        print("No matching sections or slides found.")
        return

    for result in results:
        print(f"\nID: {result['id']}")
        print(f"Course: {result['course']}")
        print(f"Title/Section: {result['title']}")
        print(f"Text: {result['snippet']}")

    print(f"\nTotal matches: {len(results)}")


if __name__ == "__main__":
    main()
