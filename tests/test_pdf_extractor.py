
from pathlib import Path
from src.extractors.pdf import extract_pdf


def test_pdf_extraction():
    project_root = Path(__file__).resolve().parent.parent
    raw_folder = project_root / "data" / "raw"

    pdf_files = list(raw_folder.rglob("*.pdf"))

    assert pdf_files, "No PDF files found in data/raw"

    pdf_file = pdf_files[0]

    results = extract_pdf(pdf_file)

    assert isinstance(results, list)
    assert len(results) > 0

    for page in results:
        assert page["location_type"] == "page"
        assert isinstance(page["page"], int)
        assert page["page"] > 0
        assert isinstance(page["text"], str)
        assert page["text"].strip()

    print(f"\nSuccessfully extracted: {pdf_file.name}")
    print(f"Pages extracted: {len(results)}")
    print(f"First page preview:\n{results[0]['text'][:300]}")
