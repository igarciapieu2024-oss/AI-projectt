import pymupdf

from src.kb import slide_tables


def _page_with_aligned_table():
    # 1. a table without lines: three hotels as columns, numbers right-aligned under each name
    document = pymupdf.open()
    page = document.new_page(width=960, height=540)
    for x, name in ((540, "San Francisco"), (680, "Chicago"), (810, "New Orleans")):
        page.insert_text((x, 150), name, fontsize=14)
    rows = (
        ("Net LT Assets", ("600.000", "1.500.000", "2.340.000")),
        ("Accum. Depreciation", ("800.000", "600.000", "390.000")),
        ("Gross LT Assets", ("1.400.000", "2.100.000", "2.730.000")),
    )
    for i, (label, values) in enumerate(rows):
        y = 200 + 36 * i
        page.insert_text((45, y), label, fontsize=14)
        for right, value in zip((624, 767, 894), values):
            width = pymupdf.get_text_length(value, fontsize=14)
            page.insert_text((right - width, y), value, fontsize=14)
    return document, page


def test_aligned_table_keeps_each_value_with_its_column():
    document, page = _page_with_aligned_table()
    text = "\n".join(slide_tables.page_tables(page))
    document.close()
    assert "Gross LT Assets: San Francisco 1.400.000; Chicago 2.100.000; New Orleans 2.730.000" in text


def test_page_without_table_gives_nothing():
    # 2. a bullet slide with one number per line is not a table
    document = pymupdf.open()
    page = document.new_page()
    for i, line in enumerate(("Inflation 2%", "Growth 1.5%", "Unemployment 11%")):
        page.insert_text((50, 100 + 30 * i), line, fontsize=14)
    assert slide_tables.page_tables(page) == []
    document.close()


def test_ruled_rendering_names_every_cell():
    # 3. two-column tables without row labels keep both headers
    text = slide_tables._render_ruled(["Most Important", "Less Important"], [["Persistence", "Teamwork"]])
    assert text == "Most Important: Persistence; Less Important: Teamwork"
