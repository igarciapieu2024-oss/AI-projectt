# Check the evaluation question bank (eval/questions.csv) before using it, so a typo in a question can't silently
# change the results. It checks the columns, the required fields, that every gold id exists in the parsed slides and
# syllabus sections, repeated or almost repeated questions, the question types, the number of questions of each type
# per author, and the dev/test split (40 dev and 60 test out of 100).
#   python -m eval.validate_questions            (add --strict to also fail on warnings)
# It only reads the files and prints the errors and warnings; it returns 1 if there are errors.

from __future__ import annotations

# argparse reads the options, csv reads the question file, json reads the parsed files and re cleans the questions.
import argparse
import csv
import json
import re

# Counter and defaultdict count the questions per author and type, and SequenceMatcher measures how similar two
# questions are.
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

# How many questions of each type every author writes (20 each). It keeps the bank balanced, so a system can't look
# strong only because one easy type of question dominates.
QUOTA = {
    "admin": 4,
    "cross": 3,
    "content": 5,
    "image": 2,
    "followup": 2,
    "unanswerable": 4,
}



# Read the ids (slide_id or section_id) of every record of a parsed file, so we can check the gold ids against them.
def load_ids(path: Path, key: str) -> set[str]:

    # The ids found in the file.
    ids = set()

    with path.open(encoding="utf-8") as file:

        for number, line in enumerate(file, start=1):

            # Empty lines are skipped.
            if not line.strip():
                continue

            # Every line has to be valid JSON; the error says which file and line failed.
            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}:{number}: {exc}"
                ) from exc

            # Every record needs its id.
            identifier = record.get(key)

            if not isinstance(identifier, str) or not identifier:
                raise ValueError(
                    f"Missing {key} at {path}:{number}"
                )

            ids.add(identifier)

    return ids


# Put the question in lower case and keep only letters and numbers, so two questions that differ only in punctuation
# or spaces are compared as the same text.
def normalized(value: str) -> str:

    # Lower case, everything that isn't a letter or a digit becomes a space, and repeated spaces are collapsed.
    value = value.lower()

    value = re.sub(r"[^a-z0-9]+", " ", value)

    return re.sub(r"\s+", " ", value).strip()


# Run every check and print the result. Errors make the file unusable; warnings are things to look at (an unfinished
# bank, similar questions). It returns 1 if there are errors (or warnings with strict), and 0 otherwise.
def validate(
    questions: Path,
    slides: Path,
    syllabus: Path,
    strict: bool = False,
) -> int:

    # The problems we find: errors and warnings.
    errors = []
    warnings = []

    # Open the question file (utf-8-sig also accepts files saved by Excel) and read its rows.
    with questions.open(
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        # The header has to be exactly the columns above, in the same order.
        if reader.fieldnames != COLUMNS:
            errors.append("Incorrect CSV columns")

        rows = list(reader)

    # An empty file is an error.
    if not rows:
        errors.append("No questions found")

    # We check the gold ids against the ids that really exist in the parsed files, so an id that looks right but
    # doesn't exist is caught.
    source_ids = (
        load_ids(slides, "slide_id")
        | load_ids(syllabus, "section_id")
    )

    # The question ids seen so far, to find repeated ones.
    seen = set()

    # The number of questions of each type per author, for the quota check below.
    by_author = defaultdict(Counter)

    # Check each row; the row number starts at 2, because row 1 of the file is the header.
    for index, row in enumerate(rows, start=2):

        # The label of this row in the messages, and each of its fields without spaces at the ends.
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

        # Every gold id has to exist in the parsed files.
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

    # Near-duplicate questions: we compare every pair, and report a similarity of 0.91 or more.

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

        # How many questions this author wrote in total.
        total = sum(counts.values())

        if total != 20:
            warnings.append(
                f"{author}: {total} questions, expected 20"
            )

        # And how many of each type.
        for qtype, expected in QUOTA.items():

            if counts[qtype] != expected:
                warnings.append(
                    f"{author}: {qtype}={counts[qtype]}, "
                    f"expected={expected}"
                )

    # The dev/test split: until the 100 questions have a split it's only a warning; after that it has to be 40 dev and
    # 60 test.

    # How many questions are in each split; a question without a split counts as "unassigned".
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

    # Every error and every warning on its own line.
    for error in errors:
        print("ERROR:", error)

    for warning in warnings:
        print("WARNING:", warning)

    print("===========================================\n")

    # The result: 1 if there are errors (or warnings with --strict), 0 if everything is fine.
    if errors:
        return 1

    if strict and warnings:
        return 1

    return 0


# Command line: the paths of the question file and of the parsed slides and syllabus sections.
def main() -> None:

    # The options: the three files and --strict.
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

    # A missing or broken file stops with a short message instead of a long error trace.
    try:
        result = validate(
            args.questions,
            args.slides,
            args.syllabus,
            args.strict,
        )

    except (OSError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")

    # Leave with the result as the exit code.
    raise SystemExit(result)


# Runs only when the file is executed directly.
if __name__ == "__main__":
    main()
