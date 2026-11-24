# Read the text written inside each slide image (OCR) with EasyOCR. run_vision.py calls ocr_image() for every image
# listed in images_index.csv, and the result is stored as the image's ocr_text in data/image_texts.jsonl.
from __future__ import annotations

# lru_cache keeps the OCR reader in memory, so it's only created once.
from functools import lru_cache
from pathlib import Path


# Create the EasyOCR reader only once. Loading its model is slow, so the cache gives back the same reader for every
# image instead of loading it again each time. We import easyocr here so the rest of the project works without it
# installed. gpu=False runs it on the CPU, so it works the same on every laptop.
@lru_cache(maxsize=1)
def _get_reader():
    import easyocr

    return easyocr.Reader(["en"], gpu=False)


# Return the English text found in one image, one paragraph per line.
def ocr_image(path: str) -> str:
    # The image has to exist.
    image_path = Path(path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # detail=0 returns only the text, without positions or confidence scores;
    # paragraph=True joins nearby lines into paragraphs instead of separate words.
    results = _get_reader().readtext(
        str(image_path),
        detail=0,
        paragraph=True,
    )

    # We drop empty results, so an image without text gives an empty string.
    return "\n".join(str(text).strip() for text in results if str(text).strip())
