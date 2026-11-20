# Read and write the shared files of the project (JSONL and CSV).
# Every step uses these functions instead of its own readers, so all files are
# written and read the same way (UTF-8, one record per line or row).
from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Type, TypeVar

T = TypeVar("T")  # the record class returned by read_jsonl


# Turn a row into a dictionary, whether it is a record from src/schemas.py
# (a dataclass) or already a dictionary.
def _as_dict(row: Any) -> dict:
    if is_dataclass(row):
        return asdict(row)
    return dict(row)


# Read a JSONL file (one JSON object per line). If a record class is given, each
# line is turned into that record, which also checks that its fields are the expected
# ones (see Record.from_dict in src/schemas.py); otherwise plain dictionaries are returned.
def read_jsonl(path: str | Path, record: Type[T] | None = None) -> list[T] | list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # skip blank lines, e.g. at the end of the file
                data = json.loads(line)
                rows.append(record.from_dict(data) if record else data)  # type: ignore[attr-defined]
    return rows


# Write rows to a JSONL file, replacing it, and return how many were written.
# The folder is created if it does not exist yet.
def write_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            # ensure_ascii=False keeps accents and symbols readable instead of \u escapes.
            f.write(json.dumps(_as_dict(row), ensure_ascii=False) + "\n")
            n += 1
    return n


# Add one row at the end of a JSONL file without rewriting it, so results already
# saved are kept if the program stops halfway.
def append_jsonl(path: str | Path, row: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_as_dict(row), ensure_ascii=False) + "\n")


# Read a CSV file into a list of dictionaries, one per row, keyed by the header.
# All values are read as text.
def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# Write rows to a CSV file with the given column order, replacing it, and return
# how many rows were written.
def write_csv(path: str | Path, rows: Iterable[Any], fieldnames: list[str]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    # newline="" lets the csv module control line endings, which avoids blank
    # lines between rows on Windows.
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_as_dict(row))
            n += 1
    return n
