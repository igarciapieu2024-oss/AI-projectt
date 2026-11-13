from src.kb import chunking


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
