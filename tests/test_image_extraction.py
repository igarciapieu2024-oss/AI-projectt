
import pymupdf

from src.visual.image_extraction import extract_pdf_images


def test_pdf_image_extraction(tmp_path):

    # Create a temporary PDF containing an image.
    image = tmp_path / "sample.png"

    from PIL import Image

    Image.new(
        "RGB",
        (200, 100),
        color="white"
    ).save(image)

    pdf_path = tmp_path / "sample.pdf"

    document = pymupdf.open()
    page = document.new_page()

    page.insert_image(
        pymupdf.Rect(50, 50, 250, 150),
        filename=str(image)
    )

    document.save(pdf_path)
    document.close()

    # Extract images.
    output_dir = tmp_path / "extracted"

    results = extract_pdf_images(
        pdf_path,
        output_dir
    )

    assert len(results) > 0
    assert results[0]["page"] == 1
    assert results[0]["source_file"] == "sample.pdf"

    assert output_dir.exists()

    extracted_path = results[0]["image_path"]

    assert image_path_exists(extracted_path)

    print("\nImage extraction successful!")
    print(results)


def image_path_exists(path):
    from pathlib import Path
    return Path(path).exists()
