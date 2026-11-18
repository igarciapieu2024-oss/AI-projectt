"""
Validate the RAG evaluation question bank.

Checks:
- CSV format
- Required fields
- Gold IDs
- Duplicate questions
- Question types
- Author quotas
- Dev/test split

Owner: Elena
"""

from __future__ import annotations

import argparse
import csv
import json
import re

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


COLUMNS = [
    "question_id",
    "split",
    "type",
    "turn1",
    "question",
    "course_code",
    "gold_ids",
    "reference_answer",
    "key_facts",
    "answerable",
    "author",
]

TYPES = {
    "content",
    "admin",
    "cross",
    "image",
    "followup",
    "unanswerable",
}

QUOTA = {
    "admin": 4,
    "cross": 3,
    "content": 5,
    "image": 2,
    "followup": 2,
    "unanswerable": 4,
}

# Per-author category targets keep the question bank balanced, so a model
# cannot appear strong simply because one easy question type dominates.


def load_ids(path: Path, key: str) -> set[str]:
    """Collect valid source identifiers from a JSONL knowledge-base file."""

    ids = set()

    with path.open(encoding="utf-8") as file:

        for number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}:{number}: {exc}"
                ) from exc

            identifier = record.get(key)

            if not isinstance(identifier, str) or not identifier:
                raise ValueError(
                    f"Missing {key} at {path}:{number}"
                )

            ids.add(identifier)

    return ids


def normalized(value: str) -> str:
    """Canonicalize question text for duplicate and similarity checks."""

    value = value.lower()

    value = re.sub(r"[^a-z0-9]+", " ", value)

    return re.sub(r"\s+", " ", value).strip()


def validate(
    questions: Path,
    slides: Path,
    syllabus: Path,
    strict: bool = False,
) -> int:
    """Check schema, ground-truth IDs, duplicates, quotas, and split rules.

    Returns the validation status used by the command-line entry point;
    warnings and errors are tracked separately.
    """

    errors = []
    warnings = []

    with questions.open(
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames != COLUMNS:
            errors.append("Incorrect CSV columns")

        rows = list(reader)

    if not rows:
        errors.append("No questions found")

    # Validate annotations against actual parsed source IDs, rather than
    # accepting plausible-looking but nonexistent slide or section labels.
    source_ids = (
        load_ids(slides, "slide_id")
        | load_ids(syllabus, "section_id")
    )

    seen = set()

    by_author = defaultdict(Counter)

    for index, row in enumerate(rows, start=2):

        label = f"Row {index}: {row.get('question_id', '?')}"

        qid = (row.get("question_id") or "").strip()

        qtype = (row.get("type") or "").strip()

        author = (row.get("author") or "").strip()

        answerable = (row.get("answerable") or "").strip()

        split = (row.get("split") or "").strip().lower()

        question = (row.get("question") or "").strip()

        reference = (row.get("reference_answer") or "").strip()

        gold = [
            item.strip()
            for item in (row.get("gold_ids") or "").split(";")
            if item.strip()
        ]

        # Question ID

        if not qid:
            errors.append(f"{label}: missing question ID")

        elif qid in seen:
            errors.append(f"{label}: duplicate question ID")

        seen.add(qid)

        # Question text

        if not question:
            errors.append(f"{label}: missing question")

        # Question type

        if qtype not in TYPES:
            errors.append(f"{label}: invalid type")

        # Author

        if not author:
            errors.append(f"{label}: missing author")

        else:
            by_author[author][qtype] += 1

        # Split

        if split not in {"", "dev", "test"}:
            errors.append(f"{label}: invalid split")

        # Answerable

        if answerable not in {"0", "1"}:
            errors.append(f"{label}: invalid answerable value")

        if qtype == "unanswerable" and answerable != "0":
            errors.append(
                f"{label}: unanswerable type must have answerable=0"
            )

        # Gold IDs

        if answerable == "0" and gold:
            errors.append(
                f"{label}: unanswerable question has gold IDs"
            )

        if answerable == "1" and not gold:
            errors.append(
                f"{label}: answerable question missing gold IDs"
            )

        for source_id in gold:

            if source_id not in source_ids:
                errors.append(
                    f"{label}: unknown gold ID {source_id}"
                )

        # Reference answer

        if answerable == "1" and not reference:
            errors.append(
                f"{label}: missing reference answer"
            )

    # Detect near-duplicate questions

    for i, left in enumerate(rows):

        for right in rows[i + 1:]:

            q1 = normalized(left.get("question") or "")

            q2 = normalized(right.get("question") or "")

            if q1 and q2:

                similarity = SequenceMatcher(
                    None, q1, q2
                ).ratio()

                if similarity >= 0.91:
                    warnings.append(
                        "Similar questions: "
                        f"{left['question_id']} / "
                        f"{right['question_id']}"
                    )

    # Check author quotas

    for author, counts in sorted(by_author.items()):

        total = sum(counts.values())

        if total != 20:
            warnings.append(
                f"{author}: {total} questions, expected 20"
            )

        for qtype, expected in QUOTA.items():

            if counts[qtype] != expected:
                warnings.append(
                    f"{author}: {qtype}={counts[qtype]}, "
                    f"expected={expected}"
                )

    # Check dev/test split

    splits = Counter(
        (row.get("split") or "").strip().lower()
        or "unassigned"
        for row in rows
    )

    if len(rows) != 100 or splits["unassigned"]:

        warnings.append(
            "Test set not locked: "
            "waiting for 100 questions and final split"
        )

    elif splits["dev"] != 40 or splits["test"] != 60:

        errors.append(
            "Expected 40 dev questions and 60 test questions"
        )

    # Results

    print("\n========== EVALUATION VALIDATION ==========")

    print(f"Total questions: {len(rows)}")

    print(f"Available source IDs: {len(source_ids)}")

    print(f"Authors: {dict(Counter(row.get('author') for row in rows))}")

    print(f"Splits: {dict(splits)}")

    print(f"Errors: {len(errors)}")

    print(f"Warnings: {len(warnings)}")

    for error in errors:
        print("ERROR:", error)

    for warning in warnings:
        print("WARNING:", warning)

    print("===========================================\n")

    if errors:
        return 1

    if strict and warnings:
        return 1

    return 0


def main() -> None:
    """Run question-bank validation from command-line path arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("eval/questions.csv"),
    )

    parser.add_argument(
        "--slides",
        type=Path,
        default=Path("data/interim/slides.jsonl"),
    )

    parser.add_argument(
        "--syllabus",
        type=Path,
        default=Path("data/interim/syllabus_sections.jsonl"),
    )

    parser.add_argument(
        "--strict",
        action="store_true",
    )

    args = parser.parse_args()

    try:
        result = validate(
            args.questions,
            args.slides,
            args.syllabus,
            args.strict,
        )

    except (OSError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")

    raise SystemExit(result)


if __name__ == "__main__":
    main()
