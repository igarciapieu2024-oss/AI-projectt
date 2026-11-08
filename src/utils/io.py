"""Read and write the shared files. Use these instead of writing your own readers."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Type, TypeVar

T = TypeVar("T")


def _as_dict(row: Any) -> dict:
    if is_dataclass(row):
        return asdict(row)
    return dict(row)


def read_jsonl(path: str | Path, record: Type[T] | None = None) -> list[T] | list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                rows.append(record.from_dict(data) if record else data)  # type: ignore[attr-defined]
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_as_dict(row), ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, row: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_as_dict(row), ensure_ascii=False) + "\n")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: Iterable[Any], fieldnames: list[str]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_as_dict(row))
            n += 1
    return n
