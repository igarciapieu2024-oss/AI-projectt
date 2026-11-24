# Find the ids of the syllabus sections, slides and slide images that contain some words. We use it when writing the
# evaluation questions, to fill in gold_ids (the correct source of each question).
#   python -m eval.find_ids "final exam"
from __future__ import annotations

# argparse reads the words to search, and json reads the records.
import argparse
import json
from pathlib import Path

# Where the parsed files are.
from src.config import PATHS


# Return every section, slide and image record that contains all the words of the query. It's a plain word search to
# help us write the questions, not the search of the system. An image match gives the id of its slide, because the
# slide is the source the knowledge base cites.
def find_ids(query: str, data_dir: Path | None = None) -> list[dict]:
    # The words of the query, in lower case.
    words = query.casefold().split()
    if not words:
        return []

    # The parsed files of the project, or the same three files in another folder (used by the tests).
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

    # The records that match.
    matches = []

    for path, record_type in files:
        # A file that doesn't exist yet is skipped with a message.
        if not path.exists():
            print(f"File not found: {path}")
            continue

        # Read the file line by line (one JSON record per line) and take the id, the course, the title and the text of
        # each record.
        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)

                # A syllabus section: its id, course, name and text.
                if record_type == "section":
                    source_id = record.get("section_id", "")
                    course = record.get("course_code", "")
                    title = record.get("section_name", "")
                    text = record.get("text", "")
                # A slide: its id, course, title and text.
                elif record_type == "slide":
                    source_id = record.get("slide_id", "")
                    course = record.get("course_code", "")
                    title = record.get("title", "")
                    text = record.get("text", "")
                # An image: the id of its slide, and its OCR text and description as the text.
                else:
                    source_id = record.get("slide_id", "")
                    course = record.get("course_code", "")
                    title = record.get("image_class", "Image")
                    text = (
                        record.get("ocr_text", "")
                        + " "
                        + record.get("description", "")
                    )

                # A record matches when every word of the query appears in its title or its text (in any case).
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


# Command line: search the words and print each match with a short piece of its text.
def main() -> None:
    # The only argument is the words to search.
    parser = argparse.ArgumentParser(
        description="Find syllabus section and slide IDs."
    )
    parser.add_argument("query", help="Words to search for")
    args = parser.parse_args()

    # Search, and print a message if nothing was found.
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


# Runs only when the file is executed directly.
if __name__ == "__main__":
    main()
