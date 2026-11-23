# Check the evaluation question bank (eval/questions.csv) before using it, so a typo in a question cannot
# silently change the results. It checks the columns, the required fields, that every gold id exists in the
# parsed slides and syllabus sections, repeated or almost repeated questions, the question types, the number
# of questions of each type per author, and the dev/test split (40 dev and 60 test out of 100).
#   python -m eval.validate_questions            (add --strict to also fail on warnings)
# It only reads the files and prints the errors and warnings; it returns 1 if there are errors.

from __future__ import annotations

import argparse
import csv
import json
import re

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


# The columns of the question file, in this order.
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

# The question types: syllabus content, admin rules, across courses, answered by an image, follow-up of a
# previous question (turn1), and questions the documents cannot answer.
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

# How many questions of each type every author writes (20 each). It keeps the bank balanced, so a system
# cannot look strong only because one easy type of question dominates.


# Read the ids (slide_id or section_id) of every record of a parsed file, to check the gold ids against them.
def load_ids(path: Path, key: str) -> set[str]:

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


# Lowercase the question and keep only letters and numbers, so two questions that differ only in punctuation
# or spaces are compared as the same text.
def normalized(value: str) -> str:

    value = value.lower()

    value = re.sub(r"[^a-z0-9]+", " ", value)

    return re.sub(r"\s+", " ", value).strip()


# Run every check and print the result. Errors make the file unusable; warnings are things to look at (an
# unfinished bank, similar questions). Returns 1 if there are errors (or warnings with strict), else 0.
def validate(
    questions: Path,
    slides: Path,
    syllabus: Path,
    strict: bool = False,
) -> int:

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

    # The gold ids are checked against the ids that really exist in the parsed files, so an id that looks right
    # but does not exist is caught.
    source_ids = (
        load_ids(slides, "slide_id")
        | load_ids(syllabus, "section_id")
    )

    seen = set()

    # Number of questions of each type per author, for the quota check below.
    by_author = defaultdict(Counter)

    # Check each row; the row number starts at 2 because row 1 of the file is the header.
    for index, row in enumerate(rows, start=2):

        label = f"Row {index}: {row.get('question_id', '?')}"

        qid = (row.get("question_id") or "").strip()

        qtype = (row.get("type") or "").strip()

        author = (row.get("author") or "").strip()

        answerable = (row.get("answerable") or "").strip()

        split = (row.get("split") or "").strip().lower()

        question = (row.get("question") or "").strip()

        reference = (row.get("reference_answer") or "").strip()

        # gold_ids can hold several ids separated by ";".
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

        # Answerable: 1 if the documents answer it, 0 if the system should refuse.

        if answerable not in {"0", "1"}:
            errors.append(f"{label}: invalid answerable value")

        if qtype == "unanswerable" and answerable != "0":
            errors.append(
                f"{label}: unanswerable type must have answerable=0"
            )

        # Gold IDs: an answerable question needs at least one, an unanswerable one has none.

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

    # Near-duplicate questions: every pair is compared, and a similarity of 0.91 or more is reported.

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

    # Author quotas: 20 questions per author, with the numbers of each type in QUOTA.

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

    # Dev/test split: until the 100 questions have a split it is only a warning; after that it must be 40 dev
    # and 60 test.

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


# Command line: the paths of the question file and of the parsed slides and syllabus sections.
def main() -> None:

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

    # A missing or broken file stops with a short message instead of a traceback.
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


# Runs only when the file is executed directly.
if __name__ == "__main__":
    main()
