
from pptx import Presentation

from src.extractors.powerpoint import extract_pptx


def test_pptx_extraction(tmp_path):

    # Create a temporary PowerPoint presentation.
    presentation = Presentation()

    slide = presentation.slides.add_slide(
        presentation.slide_layouts[1]
    )

    slide.shapes.title.text = "Machine Learning"

    slide.placeholders[1].text = (
        "Supervised learning uses labeled data."
    )

    file_path = tmp_path / "test_presentation.pptx"
    presentation.save(file_path)

    # Extract the presentation.
    results = extract_pptx(file_path)

    # Validate extraction.
    assert len(results) == 1
    assert results[0]["location_type"] == "slide"
    assert results[0]["page"] == 1

    assert "Machine Learning" in results[0]["text"]
    assert "Supervised learning" in results[0]["text"]

    print("\nPowerPoint extraction successful!")
    print(results[0]["text"])
