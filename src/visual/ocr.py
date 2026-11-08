
from pathlib import Path
import pytesseract
from PIL import Image


def extract_image_text(
    image_path: str | Path,
    language: str = "eng+spa"
) -> dict:
    """Extract text from an image using Tesseract OCR."""

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Use Tesseract installed on Windows.
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    with Image.open(image_path) as image:
        text = pytesseract.image_to_string(
            image,
            lang=language
        )

    return {
        "image_path": str(image_path),
        "content_type": "ocr_text",
        "text": text.strip()
    }
