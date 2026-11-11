
from PIL import Image, ImageDraw, ImageFont

from src.visual.ocr import extract_image_text


def test_ocr_extraction(tmp_path):

    # Create an image containing readable text.
    image = Image.new(
        "RGB",
        (1200, 300),
        color="white"
    )

    draw = ImageDraw.Draw(image)

    draw.text(
        (40, 80),
        "Machine Learning improves data analysis.",
        fill="black"
    )

    image_path = tmp_path / "ocr_test.png"
    image.save(image_path)

    # Run OCR.
    result = extract_image_text(image_path)

    assert result["content_type"] == "ocr_text"
    assert "Machine Learning" in result["text"]
    assert "data analysis" in result["text"]

    print("\nOCR extraction successful!")
    print(result["text"])