# Run OCR on every slide image listed in data/interim/images_index.csv (written by extract.py) and save the text
# of each image in data/image_texts.jsonl, one row per image. describe.py then adds the class and the description
# of each image to the same rows, and chunking.py adds that text to the chunk of its slide.
#   python -m src.vision.run_vision
# The image step takes hours, so it runs once and separately from src/pipeline.py, and every shortcut below
# saves OCR runs: images already in the output are skipped (a stopped run continues where it ended), an
# image repeated on several slides is read once, and images too small to hold text are not read.

from __future__ import annotations

import hashlib
from pathlib import Path

from src.config import PATHS, load_config
from src.schemas import ImageText
from src.utils.io import append_jsonl, read_csv, read_jsonl
from src.vision.ocr import ocr_image


# Images below these sizes (pixels) are treated as icons or logos: they are kept in the output with an empty text.
MIN_WIDTH = 120
MIN_HEIGHT = 60
MIN_AREA = 12_000


# A fingerprint of the image file (SHA-256 of its bytes): two images with the same fingerprint are the same image,
# for example a logo repeated on every slide, so its OCR text can be reused.
def _image_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# True if the image is too small to hold useful text (narrower, lower or with less area than the limits above).
def _is_too_small(row: dict) -> bool:
    width = int(row.get("width", 0))
    height = int(row.get("height", 0))

    return (
        width < MIN_WIDTH
        or height < MIN_HEIGHT
        or width * height < MIN_AREA
    )


# Read every pending image and add its row to data/image_texts.jsonl. Returns the number of rows added.
def run_vision(cfg=None) -> int:

    root = Path(PATHS["root"])
    index_path = Path(PATHS["images_index"])
    output_path = Path(PATHS["image_texts"])

    if not index_path.is_file():
        raise FileNotFoundError(f"{index_path} not found.")

    rows = read_csv(index_path)

    # The rows already saved by a previous run: their images are not read again.
    existing_records = []
    if output_path.is_file():
        existing_records = read_jsonl(output_path, ImageText)

    existing_ids = {record.image_id for record in existing_records}

    # Fingerprint -> OCR text of the images already read, so a repeated image found below reuses the text of the
    # first copy instead of running OCR again. Images with an empty text are left out of it.
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

        # Already in the output from a previous run.
        if iid in existing_ids:
            continue

        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = root / image_path

        if not image_path.is_file():
            print(f"Warning: image not found: {image_path}")
            continue

        # The text stays empty for small images; otherwise it is reused from an identical image or read with OCR.
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

        # ImageText (src/schemas.py) checks the row before it is saved; class and description are added by describe.py.
        record = ImageText(
            image_id=iid,
            slide_id=row["slide_id"],
            source_id=row["source_id"],
            kind=row["kind"],
            image_path=row["image_path"],
            ocr_text=ocr_text,
        )

        # Saved at once, so the work done is kept if the run stops.
        append_jsonl(output_path, record)
        existing_ids.add(iid)
        added += 1

        # Progress every 25 new rows, because the whole run takes a long time.
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


# Runs only when the file is executed directly.
if __name__ == "__main__":
    run_vision(load_config("configs/base.yaml"))
