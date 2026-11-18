"""Read the tables of slide pages row by row.

Output: data/interim/slide_tables.jsonl, one record per slide page that has a table:
    {"slide_id": ..., "tables": ["<one line per row>", ...]}

1. The plain text of a table slide mixes its columns: the Management Control hotel table reads
   "Net LT Assets 600.000 1.500.000 2.340.000", and the model gave 2,340,000 (Chicago) for New Orleans.
   Each row is written here with its column header next to every value:
   "Net LT Assets (end 2012): San Francisco Hotel 600.000; Chicago Hotel 1.500.000; New Orleans Hotel 2.340.000".
2. The file is separate from slides.jsonl, and the chunks use it only with kb.slide_tables, so the knowledge
   bases of earlier configurations are rebuilt exactly as before.
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from src.config import PATHS
from src.utils.io import read_csv, write_jsonl

_NUMBER = re.compile(r"^[-+(]?[$€£]?\d[\d.,]*%?\)?$|^no$|^copy$|^-$")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _render(header: list[str], rows: list[list[str]]) -> str:
    """3. One line per row: the row label, then every value with its column header."""
    lines = []
    for row in rows:
        label, values = row[0], row[1:]
        cells = [
            f"{header[i + 1]} {value}".strip() if i + 1 < len(header) and header[i + 1] else value
            for i, value in enumerate(values)
            if value
        ]
        if label and cells:
            lines.append(f"{label}: " + "; ".join(cells))
    return "\n".join(lines)


def _render_ruled(header: list[str], rows: list[list[str]]) -> str:
    """4. Ruled tables: every cell with its column header ("Most Important: Persistence; Less Important: ...")."""
    lines = []
    for row in rows:
        cells = [
            f"{header[i]}: {value}" if i < len(header) and header[i] else value
            for i, value in enumerate(row)
            if value
        ]
        if cells:
            lines.append("; ".join(cells))
    return "\n".join(lines)


def _cell_lines(page: pymupdf.Page, cell) -> list[str]:
    if not cell:
        return []
    return [_clean(line) for line in page.get_text("text", clip=cell).splitlines() if _clean(line)]


def _is_number(text: str) -> bool:
    return all(_NUMBER.match(part.lower()) for part in text.split()) if text else False


def _ruled_tables(page: pymupdf.Page) -> list[str]:
    """6. Tables drawn with lines, read cell by cell from the page (the cell text PyMuPDF returns can be split).

    A row whose cells all hold the same number of lines is several table rows drawn in one box (the EVA table
    of Management Control), so it is split back into those rows.
    """
    out = []
    try:
        # 7. newer PyMuPDF versions first run a layout model on the page, which crashed on some slides and
        #    takes seconds per page; the drawn lines are enough here
        try:
            found = page.find_tables(strategy="lines", use_layout=False).tables
        except TypeError:
            found = page.find_tables(strategy="lines").tables
    except Exception:  # noqa: BLE001 - a page PyMuPDF cannot analyse just has no ruled table
        return out
    for table in found:
        grid: list[list[str]] = []
        for row in table.rows:
            cells = [_cell_lines(page, cell) for cell in row.cells]
            counts = {len(c) for c in cells if c}
            if len(counts) == 1 and counts != {1} and sum(bool(c) for c in cells) >= 2:
                n = counts.pop()
                grid.extend([[c[i] if c else "" for c in cells] for i in range(n)])
            else:
                grid.append([" ".join(c) for c in cells])
        grid = [row for row in grid if any(row)]
        # the header is the first row with at least two non-numeric cells after the first column
        start = next(
            (i for i, row in enumerate(grid) if sum(bool(c) and not _is_number(c) for c in row[1:]) >= 1
             and sum(bool(c) for c in row) >= 2),
            None,
        )
        if start is None or len(grid) - start < 3:
            continue
        header, rows = grid[start], grid[start + 1:]
        labels = [row[0] for row in rows if row and row[0]]
        if labels and sum(not _is_number(label) for label in labels) >= len(labels) / 2 and len(header) > 2:
            text = _render(header, rows)
        else:
            text = _render_ruled(header, rows)
        if text:
            out.append(text)
    return out


def _aligned_tables(page: pymupdf.Page) -> list[str]:
    """5. Tables without lines: numbers that end at the same x position on at least three lines form a column."""
    words = page.get_text("words")
    lines: dict[int, list] = {}
    for word in words:
        lines.setdefault(round(word[1] / 6), []).append(word)
    candidates = []
    for key in sorted(lines):
        items = sorted(lines[key], key=lambda w: w[0])
        numbers = [w for w in items if _NUMBER.match(w[4].lower())]
        if len(numbers) >= 2:
            candidates.append((items, numbers))
    if len(candidates) < 3:
        return []
    edges: list[float] = []
    for _, numbers in candidates:
        for w in numbers:
            if not any(abs(w[2] - e) < 15 for e in edges):
                edges.append(w[2])
    columns = sorted(
        e for e in edges if sum(any(abs(w[2] - e) < 15 for w in numbers) for _, numbers in candidates) >= 3
    )
    if len(columns) < 2:
        return []

    def column_of(word) -> int | None:
        i = min(range(len(columns)), key=lambda c: abs(columns[c] - word[2]))
        return i if abs(columns[i] - word[2]) < 15 and _NUMBER.match(word[4].lower()) else None

    grid, spans = [], [[e, e] for e in columns]
    for items, _ in candidates:
        values = [""] * len(columns)
        first = None
        for w in items:
            i = column_of(w)
            if i is not None:
                values[i] = _clean(f"{values[i]} {w[4]}")
                spans[i][0] = min(spans[i][0], w[0])
                first = w[0] if first is None else min(first, w[0])
        if first is None or sum(bool(v) for v in values) < 2:
            continue
        label = _clean(" ".join(w[4] for w in items if w[2] <= first + 1 and column_of(w) is None))
        if label:
            grid.append((items[0][1], [label] + values))
    if len(grid) < 3:
        return []
    top = min(y for y, _ in grid)
    # header words sit just above the first row, each inside the horizontal span of one column
    header = [""] * len(columns)
    for w in sorted(words, key=lambda w: (round(w[1]), w[0])):
        if w[3] <= top + 1 and top - w[3] < 110:
            centre = (w[0] + w[2]) / 2
            for i, (lo, hi) in enumerate(spans):
                if lo - 45 <= centre <= hi + 15:
                    header[i] = _clean(f"{header[i]} {w[4]}")
                    break
    return [_render([""] + header, [row for _, row in grid])]


def page_tables(page: pymupdf.Page) -> list[str]:
    """Tables of one page: ruled tables first, aligned number tables only if there is no ruled one."""
    return _ruled_tables(page) or _aligned_tables(page)


def build_slide_tables(cfg) -> int:
    """Read every slide PDF of the manifest and write the pages that have a table."""
    del cfg
    raw_root = Path(PATHS["root"]) / "data" / "raw"
    records = []
    for row in read_csv(Path(PATHS["manifest"])):
        if row.get("doc_type") != "slides":
            continue
        with pymupdf.open(raw_root / row["file"]) as document:
            for index, page in enumerate(document):
                tables = page_tables(page)
                if tables:
                    records.append({"slide_id": f"{row['source_id']}_p{index + 1:03d}", "tables": tables})
    write_jsonl(Path(PATHS["interim"]) / "slide_tables.jsonl", records)
    return len(records)
