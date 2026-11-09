"""Run OCR over extracted images; safely append only new ImageText records."""
from __future__ import annotations

from pathlib import Path

from src.config import PATHS, load_config
from src.schemas import ImageText
from src.utils.io import append_jsonl, read_csv, read_jsonl
from src.vision.ocr import ocr_image


def run_vision(cfg=None) -> int:
    """Write one ImageText per image, skipping IDs already in the output file."""
    root = Path(PATHS["root"])
    index_path = Path(PATHS["images_index"])
    output_path = Path(PATHS["image_texts"])
    if not index_path.is_file():
        raise FileNotFoundError(
            f"{index_path} not found. Run extract_images first."
        )

    existing = set()
    if output_path.is_file():
        existing = {r.image_id for r in read_jsonl(output_path, ImageText)}

    added = 0
    for row in read_csv(index_path):
        iid = row["image_id"]
        if iid in existing:
            continue
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = root / image_path

        record = ImageText(
            image_id=iid,
            slide_id=row["slide_id"],
            source_id=row["source_id"],
            kind=row["kind"],
            image_path=row["image_path"],
            ocr_text=ocr_image(str(image_path)),
        )
        append_jsonl(output_path, record)
        existing.add(iid)
        added += 1
        if added % 10 == 0:
            print(f"OCR processed {added} new images...")

    print(f"OCR complete: {added} new records added.")
    return added


if __name__ == "__main__":
    run_vision(load_config("configs/base.yaml"))
