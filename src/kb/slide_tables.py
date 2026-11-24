# Read the tables of the slide pages row by row, keeping every value next to its column header. The plain text of a
# table slide mixes its columns: the Management Control hotel table is read as "Net LT Assets 600.000 1.500.000
# 2.340.000", and the model gave 2,340,000 (Chicago) for New Orleans. Here we write each row as one line, with its
# column header next to every value:
# "Net LT Assets (end 2012): San Francisco Hotel 600.000; Chicago Hotel 1.500.000; New Orleans Hotel 2.340.000".
# Output: data/interim/slide_tables.jsonl, one record per slide page that has a table. chunking.py adds these lines to
# the chunk of their slide, but only when kb.slide_tables is true (final_system.yaml). We keep the file separate from
# slides.jsonl, so the knowledge bases of the earlier configurations, which don't use it, are rebuilt exactly as
# before.
from __future__ import annotations

# Regular expressions, to recognise numbers and clean spaces.
import re
from pathlib import Path

# PyMuPDF reads the PDFs: their text, their words with positions and the tables drawn with lines.
import pymupdf

# The folders of the project, the CSV reader (for the manifest) and the JSONL writer.
from src.config import PATHS
from src.utils.io import read_csv, write_jsonl

# A cell that counts as a value: a number with an optional sign, currency symbol, % or brackets, or one of the words
# "no", "copy" and "-", which are written as values in some of our tables.
_NUMBER = re.compile(r"^[-+(]?[$€£]?\d[\d.,]*%?\)?$|^no$|^copy$|^-$")


# ----------------------------------------------------------------------
# Small helpers

# Collapse line breaks and repeated spaces into one space.
def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# Write a table whose first column holds the row names: one line per row, with the row name first and then every value
# with its column header. Rows without a name or without values are skipped.
    
def _render(header: list[str], rows: list[list[str]]) -> str:
    # One line of text per row of the table.
    lines = []
    for row in rows:
        # The first cell is the name of the row, and the rest are its values.
        label, values = row[0], row[1:]
        # header[i + 1] because header[0] is the title of the row-name column.
        cells = [
            f"{header[i + 1]} {value}".strip() if i + 1 < len(header) and header[i + 1] else value
            for i, value in enumerate(values)
            if value
        ]
        # A row is only written when it has a name and at least one value.
        if label and cells:
            lines.append(f"{label}: " + "; ".join(cells))
    return "\n".join(lines)


# Write a table whose first column isn't a list of row names: every cell is written as "header: value".

def _render_ruled(header: list[str], rows: list[list[str]]) -> str:
    # One line of text per row of the table.
    lines = []
    for row in rows:
        cells = [
            f"{header[i]}: {value}" if i < len(header) and header[i] else value
            for i, value in enumerate(row)
            if value
        ]
        # A row with no values is skipped.
        if cells:
            lines.append("; ".join(cells))
    return "\n".join(lines)


# The text inside one cell, read from the area of the cell on the page (clip), one item per line. We read it this way
# because the cell text PyMuPDF gives directly can come split.
def _cell_lines(page: pymupdf.Page, cell) -> list[str]:
    # An empty cell has no area.
    if not cell:
        return []
    return [_clean(line) for line in page.get_text("text", clip=cell).splitlines() if _clean(line)]


# True if every word of the text is a value (see _NUMBER); an empty text isn't a value.
def _is_number(text: str) -> bool:
    return all(_NUMBER.match(part.lower()) for part in text.split()) if text else False


# ----------------------------------------------------------------------
# Tables drawn with lines

# Find the tables drawn with lines on one page and read them cell by cell. It returns one text per table.
    
def _ruled_tables(page: pymupdf.Page) -> list[str]:
    # The texts of the tables found on this page.
    out = []
    try:
        # Newer PyMuPDF versions first run a layout model on the page, which crashed on some slides and takes seconds
        # per page; the drawn lines are enough here, so we switch it off. Older versions don't have the use_layout
        # option (TypeError), so we call them without it.
        try:
            found = page.find_tables(strategy="lines", use_layout=False).tables
        except TypeError:
            found = page.find_tables(strategy="lines").tables
    except Exception:  # noqa: BLE001 - a page PyMuPDF cannot analyse just has no ruled table
        return out
    for table in found:
        # Build the grid of the table: one list of cell texts per row.
        grid: list[list[str]] = []
        for row in table.rows:
            # The lines of text of every cell of this row.
            cells = [_cell_lines(page, cell) for cell in row.cells]
            # The number of lines in each non-empty cell. When every filled cell has the same number of lines (more
            # than one), and at least two cells are filled, the box holds several table rows drawn together (as in the
            # EVA table of Management Control), so we split it back into those rows.
            counts = {len(c) for c in cells if c}
            if len(counts) == 1 and counts != {1} and sum(bool(c) for c in cells) >= 2:
                n = counts.pop()
                grid.extend([[c[i] if c else "" for c in cells] for i in range(n)])
            else:
                grid.append([" ".join(c) for c in cells])
        # Remove the rows where every cell is empty.
        grid = [row for row in grid if any(row)]
        # The header is the first row with at least two filled cells and at least one non-numeric cell
        # after the first column (headers are names, not values).
        start = next(
            (i for i, row in enumerate(grid) if sum(bool(c) and not _is_number(c) for c in row[1:]) >= 1
             and sum(bool(c) for c in row) >= 2),
            None,
        )
        # Without a header and at least two rows under it, it isn't a real table (for example, a framed box of text).
        if start is None or len(grid) - start < 3:
            continue
        # The header row, and the rows under it.
        header, rows = grid[start], grid[start + 1:]
        # Choose the format: if at least half of the first-column cells are names (not values) and the table has more
        # than two columns, the first column holds row names and we use _render(); otherwise every cell is written
        # with its header by _render_ruled().
        labels = [row[0] for row in rows if row and row[0]]
        if labels and sum(not _is_number(label) for label in labels) >= len(labels) / 2 and len(header) > 2:
            text = _render(header, rows)
        else:
            text = _render_ruled(header, rows)
        # An empty text adds nothing.
        if text:
            out.append(text)
    return out


# ----------------------------------------------------------------------
# Tables without lines

# Find a table without drawn lines from the position of its numbers: numbers whose right edge is at the same x
# position on at least three lines form a column. It returns a list with one table text, or nothing.
def _aligned_tables(page: pymupdf.Page) -> list[str]:
    # Every word of the page with its box: (x0, y0, x1, y1, text, ...); x grows to the right, y downwards.
    words = page.get_text("words")
    # Group the words into lines by their top position. Dividing by 6 and rounding puts words that are less than about
    # 6 points apart vertically into the same line.
    lines: dict[int, list] = {}
    for word in words:
        lines.setdefault(round(word[1] / 6), []).append(word)
    # Candidate rows: lines, read left to right, with at least two values.
    candidates = []
    for key in sorted(lines):
        # The words of this line from left to right, and which of them are values.
        items = sorted(lines[key], key=lambda w: w[0])
        numbers = [w for w in items if _NUMBER.match(w[4].lower())]
        if len(numbers) >= 2:
            candidates.append((items, numbers))
    # Fewer than three such lines isn't a table.
    if len(candidates) < 3:
        return []
    # The possible column positions: the right edges (x1) of the values. Numbers are aligned on their right edge, and
    # edges less than 15 points apart count as the same column.
    edges: list[float] = []
    for _, numbers in candidates:
        for w in numbers:
            if not any(abs(w[2] - e) < 15 for e in edges):
                edges.append(w[2])
    # We keep only the positions that have a value on at least three lines, sorted from left to right.
    columns = sorted(
        e for e in edges if sum(any(abs(w[2] - e) < 15 for w in numbers) for _, numbers in candidates) >= 3
    )
    # A table needs at least two value columns.
    if len(columns) < 2:
        return []

    # The column of one word: the nearest column, only if the word is a value and is less than 15 points from it;
    # otherwise None (the word isn't in a value column).
    def column_of(word) -> int | None:
        i = min(range(len(columns)), key=lambda c: abs(columns[c] - word[2]))
        return i if abs(columns[i] - word[2]) < 15 and _NUMBER.match(word[4].lower()) else None

    # grid: (top position, [row name, value of column 1, value of column 2, ...]) for each row.
    # spans: the horizontal extent [left, right] of each column, which gets wider as values are placed in it; we use
    # it below to place the header words above the right column.
    grid, spans = [], [[e, e] for e in columns]
    for items, _ in candidates:
        # One empty value per column to start with.
        values = [""] * len(columns)
        first = None  # left edge of the first value of the row
        for w in items:
            # The column of this word, if it is a value in one of the columns.
            i = column_of(w)
            if i is not None:
                # Add the word to its column, widen that column to the left if needed and remember where the values
                # start.
                values[i] = _clean(f"{values[i]} {w[4]}")
                spans[i][0] = min(spans[i][0], w[0])
                first = w[0] if first is None else min(first, w[0])
        # A row needs values in at least two columns.
        if first is None or sum(bool(v) for v in values) < 2:
            continue
        # The row name: the words that end before the first value of the row and are not values.
        label = _clean(" ".join(w[4] for w in items if w[2] <= first + 1 and column_of(w) is None))
        if label:
            grid.append((items[0][1], [label] + values))
    # Fewer than three rows isn't a table.
    if len(grid) < 3:
        return []
    # Top of the first row of the table.
    top = min(y for y, _ in grid)
    # The header words sit just above the first row (their bottom at most 110 points above it), each one inside the
    # horizontal extent of one column; the margins (45 points to the left, 15 to the right) allow for header words
    # that are wider than the numbers under them.
    header = [""] * len(columns)
    for w in sorted(words, key=lambda w: (round(w[1]), w[0])):
        if w[3] <= top + 1 and top - w[3] < 110:
            # The horizontal centre of the word decides which column it is above.
            centre = (w[0] + w[2]) / 2
            for i, (lo, hi) in enumerate(spans):
                if lo - 45 <= centre <= hi + 15:
                    header[i] = _clean(f"{header[i]} {w[4]}")
                    break
    # The first column (the row names) has no header, which is why there is a "" in front.
    return [_render([""] + header, [row for _, row in grid])]


# ----------------------------------------------------------------------
# All the slides

# The tables of one page. We read the tables drawn with lines first, and only look for aligned numbers when the page
# has no ruled table, so the same table isn't written twice.
def page_tables(page: pymupdf.Page) -> list[str]:
    return _ruled_tables(page) or _aligned_tables(page)


# Read every slide PDF listed in the manifest and write the pages that have a table to
# data/interim/slide_tables.jsonl. It returns the number of pages with tables. The build pipeline (src/pipeline.py)
# calls it before the chunking, only when kb.slide_tables is true.
def build_slide_tables(cfg) -> int:
    # We don't need the settings: all the paths come from src.config.PATHS.
    del cfg
    # The folder with the PDFs.
    raw_root = Path(PATHS["root"]) / "data" / "raw"
    records = []
    # Go through every file listed in the manifest.
    for row in read_csv(Path(PATHS["manifest"])):
        # Only slide decks; the syllabus tables are handled in syllabus_text.py.
        if row.get("doc_type") != "slides":
            continue
        # Open the PDF and look for tables page by page.
        with pymupdf.open(raw_root / row["file"]) as document:
            for index, page in enumerate(document):
                tables = page_tables(page)
                if tables:
                    # The same slide_id format as parse_slides.py (<source_id>_p012, pages from 1), so chunking.py can
                    # join each table to its slide.
                    records.append({"slide_id": f"{row['source_id']}_p{index + 1:03d}", "tables": tables})
    # Write one record per page with tables, and return how many there are.
    write_jsonl(Path(PATHS["interim"]) / "slide_tables.jsonl", records)
    return len(records)
