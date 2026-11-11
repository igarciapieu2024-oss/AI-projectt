
from pathlib import Path
import hashlib
import pymupdf


def extract_pdf_images(
    file_path: str | Path,
    output_dir: str | Path
) -> list[dict]:
    """
    Extract embedded images from a PDF.

    Saves images separately and preserves source page numbers.
    """

    file_path = Path(file_path)
    output_dir = Path(output_dir)

    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Expected a PDF file")

    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_images = []

    with pymupdf.open(file_path) as document:

        for page_number, page in enumerate(
            document, start=1
        ):

            images = page.get_images(full=True)

            for image_index, image_info in enumerate(
                images, start=1
            ):

                xref = image_info[0]

                image_data = document.extract_image(xref)

                image_bytes = image_data["image"]
                extension = image_data["ext"]

                image_hash = hashlib.sha256(
                    image_bytes
                ).hexdigest()[:12]

                image_name = (
                    f"page_{page_number}_"
                    f"image_{image_index}_"
                    f"{image_hash}.{extension}"
                )

                image_path = output_dir / image_name

                image_path.write_bytes(image_bytes)

                extracted_images.append({
                    "source_file": file_path.name,
                    "page": page_number,
                    "image_index": image_index,
                    "image_path": str(image_path),
                    "image_hash": image_hash,
                    "extension": extension
                })

    return extracted_images
