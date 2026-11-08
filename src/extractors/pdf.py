
from pathlib import Path
import pymupdf


def extract_pdf(file_path: str | Path) -> list[dict]:
    """
    Extract text from a PDF document, page by page.

    Returns a list of dictionaries containing:
    - page number
    - location type
    - extracted text
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {file_path}")

    extracted_pages = []

    with pymupdf.open(file_path) as document:

        if document.is_encrypted:
            raise ValueError(
                f"PDF is password protected: {file_path.name}"
            )

        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text", sort=True).strip()

            # Skip pages without extractable text.
            if not text:
                continue

            extracted_pages.append({
                "location_type": "page",
                "page": page_number,
                "text": text
            })

    return extracted_pages