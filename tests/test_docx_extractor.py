
from docx import Document

from src.extractors.word import extract_docx


def test_docx_extraction(tmp_path):

    # Create a temporary Word document.
    document = Document()

    document.add_heading(
        "Business Analytics",
        level=1
    )

    document.add_paragraph(
        "Data analysis supports business decisions."
    )

    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Concept"
    table.rows[0].cells[1].text = "Description"

    file_path = tmp_path / "test_document.docx"
    document.save(file_path)

    # Extract the document.
    results = extract_docx(file_path)

    assert len(results) == 1

    text = results[0]["text"]

    assert "Business Analytics" in text
    assert "Data analysis" in text
    assert "Concept" in text
    assert "Description" in text

    assert results[0]["location_type"] == "document"

    print("\nWord extraction successful!")
    print(text)