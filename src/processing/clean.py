
import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Clean extracted document text while preserving
    meaningful paragraph breaks and content.
    """

    if not isinstance(text, str):
        return ""

    # Normalize Unicode characters.
    text = unicodedata.normalize("NFC", text)

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove non-printable control characters,
    # while preserving newlines and tabs.
    text = "".join(
        char for char in text
        if char.isprintable() or char in "\n\t"
    )

    # Replace tabs and repeated spaces.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove unnecessary spaces at line beginnings
    # and endings.
    lines = [
        line.strip()
        for line in text.split("\n")
    ]

    text = "\n".join(lines)

    # Limit consecutive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_records(records: list[dict]) -> list[dict]:
    """
    Clean extracted records while preserving metadata.

    Records without meaningful text are discarded.
    """

    cleaned_records = []

    for record in records:

        if not isinstance(record, dict):
            continue

        cleaned_record = record.copy()

        cleaned_text = clean_text(
            cleaned_record.get("text", "")
        )

        if not cleaned_text:
            continue

        cleaned_record["text"] = cleaned_text

        cleaned_records.append(cleaned_record)

    return cleaned_records