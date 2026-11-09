
"""Text written inside each image (OCR).

Owner: Santiago.
Uses EasyOCR, with CPU inference for cross-platform compatibility.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _get_reader():
    """Initialize and cache the EasyOCR reader."""
    import easyocr

    return easyocr.Reader(["en"], gpu=False)


def ocr_image(path: str) -> str:
    """Extract English text from one image using EasyOCR."""
    image_path = Path(path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = _get_reader().readtext(
        str(image_path),
        detail=0,
        paragraph=True,
    )

    return "\n".join(str(text).strip() for text in results if str(text).strip())
