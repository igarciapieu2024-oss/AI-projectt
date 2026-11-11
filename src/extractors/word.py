
from pathlib import Path
from docx import Document


def extract_docx(file_path: str | Path) -> list[dict]:
    """Extract paragraphs and tables from a Word document."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Word document not found: {file_path}"
        )

    if file_path.suffix.lower() != ".docx":
        raise ValueError(
            f"Expected a DOCX file: {file_path}"
        )

    document = Document(file_path)

    parts = []

    # Extract paragraphs.
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            parts.append(text)

    # Extract tables.
    for table in document.tables:
        for row in table.rows:
            cells = [
                cell.text.strip()
                for cell in row.cells
            ]

            row_text = " | ".join(cells)

            if row_text.strip():
                parts.append(row_text)

    text = "\n".join(parts)

    if not text.strip():
        return []

    return [{
        "location_type": "document",
        "page": None,
        "text": text
    }]
