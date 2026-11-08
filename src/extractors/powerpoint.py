
from pathlib import Path
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE


def extract_shape_text(shape):
    """Extract text from PowerPoint shapes, tables and groups."""

    parts = []

    # Handle grouped shapes recursively.
    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for child in shape.shapes:
            parts.extend(extract_shape_text(child))

    # Handle tables.
    elif shape.has_table:
        for row in shape.table.rows:
            cells = [
                cell.text.strip()
                for cell in row.cells
            ]

            parts.append(" | ".join(cells))

    # Handle regular text boxes and placeholders.
    elif shape.has_text_frame:
        text = shape.text.strip()

        if text:
            parts.append(text)

    return parts


def extract_pptx(file_path: str | Path) -> list[dict]:
    """Extract text from a PowerPoint presentation slide by slide."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"PowerPoint file not found: {file_path}"
        )

    if file_path.suffix.lower() != ".pptx":
        raise ValueError(
            f"Expected a PPTX file: {file_path}"
        )

    presentation = Presentation(file_path)
    extracted_slides = []

    for slide_number, slide in enumerate(
        presentation.slides, start=1
    ):

        parts = []

        for shape in slide.shapes:
            parts.extend(extract_shape_text(shape))

        text = "\n".join(parts).strip()

        if not text:
            continue

        extracted_slides.append({
            "location_type": "slide",
            "page": slide_number,
            "text": text
        })

    return extracted_slides
