from src.kb import chunking
from types import SimpleNamespace


def test_image_shows_filler_is_removed():
    assert chunking._strip_description_filler(
        "The image shows a line chart of sales."
    ) == "a line chart of sales."


def test_chart_shows_filler_is_removed():
    assert chunking._strip_description_filler(
        "The chart shows revenue over time."
    ) == "revenue over time."


def test_filler_removal_handles_capitalization_leading_space_and_punctuation():
    assert chunking._strip_description_filler(
        "  tHe ChArT ShOwS: — Quarterly revenue."
    ) == "Quarterly revenue."
    assert chunking._strip_description_filler(
        "\tTHE IMAGE SHOWS... A scatter plot."
    ) == "A scatter plot."


def test_frequent_generic_opening_variants_are_removed():
    assert chunking._strip_description_filler(
        "The image is a schematic of the process."
    ) == "a schematic of the process."
    assert chunking._strip_description_filler(
        "The image contains a table of values."
    ) == "a table of values."
    assert chunking._strip_description_filler(
        "The chart displays inflation by year."
    ) == "inflation by year."


def test_chart_is_filler_handles_capitalization_and_punctuation():
    assert chunking._strip_description_filler(
        "  tHe ChArT iS: — Showing annual sales."
    ) == "Showing annual sales."


def test_image_illustrates_filler_handles_capitalization_and_punctuation():
    assert chunking._strip_description_filler(
        "THE IMAGE ILLUSTRATES... A data flow."
    ) == "A data flow."


def test_description_without_filler_is_unchanged():
    description = "  A scatter plot compares two variables.  "

    assert chunking._strip_description_filler(description) == description


def test_filler_phrase_in_middle_is_unchanged():
    description = "Overview: The image shows a chart, then a table follows."

    assert chunking._strip_description_filler(description) == description


def test_ocr_is_unchanged_when_description_filler_is_removed(tmp_path, monkeypatch):
    image_texts_path = tmp_path / "image_texts.jsonl"
    image_texts_path.write_text("", encoding="utf-8")
    monkeypatch.setitem(chunking.PATHS, "image_texts", image_texts_path)
    image = {
        "image_id": "slide_p001_img01",
        "slide_id": "slide_p001",
        "source_id": "slide_source",
        "kind": "embedded",
        "image_path": "image.png",
        "ocr_text": "OCR VALUE 17% remains exactly as extracted.",
        "image_class": "chart",
        "description": "The chart shows: Revenue rises in year two.",
    }
    monkeypatch.setattr(chunking, "read_jsonl", lambda path: [image])

    assembled = chunking._load_image_text(use_ocr=True, use_captions=True)

    assert assembled["slide_p001"] == [
        "[Image: chart] OCR: OCR VALUE 17% remains exactly as extracted. "
        "Description: Revenue rises in year two."
    ]
    assert image["ocr_text"] == "OCR VALUE 17% remains exactly as extracted."


def test_shorten_urls_preserves_domain_subdomains_and_punctuation():
    text = (
        "See HTTPS://WWW.Example.com/a/b?q=x&lang=en#part, "
        "www.docs.example.org/page. (http://localhost:8888/tree)."
    )

    assert chunking._shorten_urls(text) == (
        "See source: example.com, source: docs.example.org. "
        "(source: localhost)."
    )


def test_shorten_urls_leaves_non_urls_and_email_addresses_unchanged():
    text = "Contact user@example.org; mention example.org without a URL scheme."

    assert chunking._shorten_urls(text) == text


def test_build_chunks_shortens_only_embed_text_and_preserves_source_text(tmp_path, monkeypatch):
    monkeypatch.setitem(chunking.PATHS, "manifest", tmp_path / "missing-manifest.csv")
    monkeypatch.setitem(chunking.PATHS, "slides", tmp_path / "slides.jsonl")
    monkeypatch.setattr(chunking, "_load_image_text", lambda *args, **kwargs: {})
    source_text = "Reference: https://docs.example.org/guide?page=4&lang=en."
    slide = {
        "slide_id": "course_s03_p004",
        "source_id": "course_s03",
        "course_code": "course",
        "sessions": [3],
        "page": 4,
        "title": "References",
        "outline": [],
        "text": source_text,
        "is_agenda": False,
        "is_low_text": False,
        "is_garbled": False,
        "n_images": 0,
    }
    monkeypatch.setattr(chunking, "read_jsonl", lambda path: [slide])
    written = {}
    monkeypatch.setattr(chunking, "chunks_path", lambda _: tmp_path / "chunks.jsonl")
    monkeypatch.setattr(chunking, "parents_path", lambda _: tmp_path / "parents.jsonl")
    monkeypatch.setattr(chunking, "write_jsonl", lambda path, rows: written.setdefault(path.name, list(rows)))
    cfg = SimpleNamespace(
        kb={"sources": ["slides"], "chunk_size": 800, "chunk_overlap": 120},
        kb_sig="test",
    )

    chunking.build_chunks(cfg)

    chunk = written["chunks.jsonl"][0]
    assert chunk.text == source_text
    assert "source: docs.example.org." in chunk.embed_text
    assert "?page=4&lang=en" not in chunk.embed_text
    assert chunk.chunk_id == "slide__course_s03_p004__00"
    assert chunk.parent_id == "parent__course_s03_p004"
    assert chunk.source_id == slide["source_id"]
