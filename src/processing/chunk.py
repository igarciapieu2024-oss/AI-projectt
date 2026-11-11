
import hashlib
import re


def split_text_into_chunks(
    text: str,
    chunk_size: int = 350,
    overlap: int = 50
) -> list[str]:
    """
    Split text into paragraph-aware chunks.

    Chunk size and overlap are measured in words.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size"
        )

    if not isinstance(text, str) or not text.strip():
        return []

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    chunks = []
    current_words = []

    for paragraph in paragraphs:

        words = paragraph.split()

        if not words:
            continue

        # Add the paragraph if it fits.
        if len(current_words) + len(words) <= chunk_size:
            current_words.extend(words)
            continue

        # Save the current chunk before starting another.
        if current_words:
            chunks.append(" ".join(current_words))

            # Preserve the configured overlap.
            current_words = (
                current_words[-overlap:] if overlap else []
            )

        # Split oversized paragraphs while enforcing
        # the maximum chunk size.
        combined = current_words + words

        while len(combined) > chunk_size:

            chunks.append(
                " ".join(combined[:chunk_size])
            )

            step = chunk_size - overlap
            combined = combined[step:]

        current_words = combined

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def chunk_records(
    records: list[dict],
    chunk_size: int = 350,
    overlap: int = 50
) -> list[dict]:
    """
    Split extracted records while preserving metadata.
    """

    chunked_records = []

    for record in records:

        text = record.get("text", "")

        chunks = split_text_into_chunks(
            text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for index, chunk_text in enumerate(chunks):

            chunk_record = record.copy()

            chunk_record["text"] = chunk_text
            chunk_record["chunk_index"] = index
            chunk_record["total_chunks"] = len(chunks)

            # Generate a reproducible unique identifier.
            identifier = "|".join([
                str(record.get("course_name", "")),
                str(record.get("source_file", "")),
                str(record.get("page", "")),
                str(record.get("content_type", "")),
                str(index),
                chunk_text
            ])

            chunk_record["chunk_id"] = hashlib.sha256(
                identifier.encode("utf-8")
            ).hexdigest()[:16]

            chunked_records.append(chunk_record)

    return chunked_records
