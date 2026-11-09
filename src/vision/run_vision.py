"""Run optimized OCR, reusing results for duplicate images."""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.config import PATHS, load_config
from src.schemas import ImageText
from src.utils.io import append_jsonl, read_csv, read_jsonl
from src.vision.ocr import ocr_image


# Images below these dimensions are treated as icons/logos.
MIN_WIDTH = 120
MIN_HEIGHT = 60
MIN_AREA = 12_000


def _image_hash(path: Path) -> str:
    """Return a SHA-256 hash of the image file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_too_small(row: dict) -> bool:
    """Identify images too small to contain useful text."""
    width = int(row.get("width", 0))
    height = int(row.get("height", 0))

    return (
        width < MIN_WIDTH
        or height < MIN_HEIGHT
        or width * height < MIN_AREA
    )


def run_vision(cfg=None) -> int:
    """OCR unique images, skipping small images and reusing duplicates."""

    root = Path(PATHS["root"])
    index_path = Path(PATHS["images_index"])
    output_path = Path(PATHS["image_texts"])

    if not index_path.is_file():
        raise FileNotFoundError(f"{index_path} not found.")

    rows = read_csv(index_path)

    existing_records = []
    if output_path.is_file():
        existing_records = read_jsonl(output_path, ImageText)

    existing_ids = {record.image_id for record in existing_records}

    # Recover OCR results from the previous run.
    hash_cache = {}

    for record in existing_records:
        if not record.ocr_text.strip():
            continue

        image_path = Path(record.image_path)
        if not image_path.is_absolute():
            image_path = root / image_path

        if image_path.is_file():
            try:
                hash_cache[_image_hash(image_path)] = record.ocr_text
            except OSError:
                continue

    added = 0
    skipped_small = 0
    reused = 0
    processed_unique = 0

    total = len(rows)

    print(f"Images in index: {total}")
    print(f"Existing OCR records: {len(existing_ids)}")
    print(f"Cached unique OCR results: {len(hash_cache)}")
    print("Starting optimized OCR...")

    for position, row in enumerate(rows, start=1):
        iid = row["image_id"]

        if iid in existing_ids:
            continue

        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = root / image_path

        if not image_path.is_file():
            print(f"Warning: image not found: {image_path}")
            continue

        ocr_text = ""

        if _is_too_small(row):
            skipped_small += 1

        else:
            image_hash = _image_hash(image_path)

            if image_hash in hash_cache:
                ocr_text = hash_cache[image_hash]
                reused += 1

            else:
                ocr_text = ocr_image(str(image_path))
                hash_cache[image_hash] = ocr_text
                processed_unique += 1

        record = ImageText(
            image_id=iid,
            slide_id=row["slide_id"],
            source_id=row["source_id"],
            kind=row["kind"],
            image_path=row["image_path"],
            ocr_text=ocr_text,
        )

        append_jsonl(output_path, record)
        existing_ids.add(iid)
        added += 1

        if added % 25 == 0:
            print(
                f"Progress: {added} new records | "
                f"{processed_unique} OCR runs | "
                f"{reused} duplicates reused | "
                f"{skipped_small} small images skipped"
            )

    print("\nOCR processing complete.")
    print(f"New records added: {added}")
    print(f"Unique images OCR processed: {processed_unique}")
    print(f"Duplicate images reused: {reused}")
    print(f"Small images skipped: {skipped_small}")
    print(f"Total records now: {len(existing_ids)}")

    return added


if __name__ == "__main__":
    run_vision(load_config("configs/base.yaml"))
