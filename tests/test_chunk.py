
from src.processing.chunk import (
    split_text_into_chunks,
    chunk_records
)


def test_split_text_into_chunks():

    text = " ".join(
        [f"word{i}" for i in range(100)]
    )

    chunks = split_text_into_chunks(
        text,
        chunk_size=30,
        overlap=5
    )

    assert len(chunks) > 1

    # No chunk should exceed the configured size.
    for chunk in chunks:
        assert len(chunk.split()) <= 30

    # Verify overlap between consecutive chunks.
    first_words = chunks[0].split()
    second_words = chunks[1].split()

    assert first_words[-5:] == second_words[:5]

    print("\nText chunking successful!")
    print(f"Chunks generated: {len(chunks)}")


def test_chunk_records():

    records = [
        {
            "course_name": "Statistics",
            "source_file": "Probability.pdf",
            "page": 12,
            "content_type": "text",
            "text": " ".join(
                [f"concept{i}" for i in range(100)]
            )
        }
    ]

    results = chunk_records(
        records,
        chunk_size=30,
        overlap=5
    )

    assert len(results) > 1

    assert results[0]["course_name"] == "Statistics"
    assert results[0]["source_file"] == "Probability.pdf"
    assert results[0]["page"] == 12

    assert results[0]["chunk_index"] == 0

    assert results[0]["chunk_id"]

    # All generated IDs must be unique.
    ids = [record["chunk_id"] for record in results]

    assert len(ids) == len(set(ids))

    print("\nRecord chunking successful!")
    print(f"Total chunks: {len(results)}")