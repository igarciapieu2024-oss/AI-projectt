"""Extract slide images and selected full-slide renders."""
from __future__ import annotations

import csv
import hashlib
from io import BytesIO
from pathlib import Path

import pymupdf
from PIL import Image

from src.config import PATHS
from src.schemas import SlideRecord
from src.utils.ids import image_id as make_image_id, slide_id as make_slide_id
from src.utils.io import read_csv, read_jsonl

INDEX_FIELDS = ["image_id", "slide_id", "source_id", "kind",
                "image_path", "thumb_path", "width", "height"]
MIN_WIDTH, MIN_HEIGHT, MIN_PIXELS = 80, 80, 8000
THUMB_SIZE = (480, 360)


def _relative(path: Path) -> str:
    return path.relative_to(Path(PATHS["root"])).as_posix()


def _thumbnail(image_path: Path, thumb_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        size = im.size
        im.thumbnail(THUMB_SIZE)
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(thumb_path, "JPEG", quality=85, optimize=True)
    return size


def extract_images(cfg=None) -> int:
    """Extract unique embedded images and render low-text/garbled slides."""
    root = Path(PATHS["root"])
    raw_root = root / "data" / "raw"
    images_dir, thumbs_dir = Path(PATHS["images"]), Path(PATHS["thumbs"])
    images_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_csv(PATHS["manifest"])
    slides_file = Path(PATHS["slides"])
    slide_rows = read_jsonl(slides_file, SlideRecord) if slides_file.is_file() else []
    flagged = {(s.source_id, s.page) for s in slide_rows
               if s.is_low_text or s.is_garbled}

    records = []
    seen_hashes = set()

    for source in manifest:
        if source.get("doc_type") != "slides":
            continue
        source_id = source["source_id"]
        pdf_path = raw_root / source["file"]
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF missing for {source_id}: {pdf_path}")

        with pymupdf.open(pdf_path) as pdf:
            for page_no, page in enumerate(pdf, start=1):
                sid = make_slide_id(source_id, page_no)
                image_no = 0

                for info in page.get_images(full=True):
                    extracted = pdf.extract_image(info[0])
                    data = extracted.get("image", b"")
                    if not data:
                        continue
                    digest = hashlib.sha256(data).hexdigest()
                    if digest in seen_hashes:
                        continue
                    try:
                        with Image.open(BytesIO(data)) as im:
                            width, height = im.size
                            im.verify()
                    except Exception:
                        continue
                    if (width < MIN_WIDTH or height < MIN_HEIGHT
                            or width * height < MIN_PIXELS):
                        continue

                    seen_hashes.add(digest)
                    image_no += 1
                    iid = make_image_id(sid, image_no)
                    ext = extracted.get("ext", "png").lower()
                    if ext == "jpeg":
                        ext = "jpg"
                    if ext not in {"png", "jpg", "webp"}:
                        ext = "png"
                    image_path = images_dir / f"{iid}.{ext}"
                    thumb_path = thumbs_dir / f"{iid}.jpg"
                    if not image_path.exists():
                        image_path.write_bytes(data)
                    width, height = _thumbnail(image_path, thumb_path)
                    records.append({
                        "image_id": iid, "slide_id": sid,
                        "source_id": source_id, "kind": "embedded",
                        "image_path": _relative(image_path),
                        "thumb_path": _relative(thumb_path),
                        "width": width, "height": height,
                    })

                if (source_id, page_no) in flagged:
                    iid = make_image_id(sid)
                    image_path = images_dir / f"{iid}.png"
                    thumb_path = thumbs_dir / f"{iid}.jpg"
                    if not image_path.exists():
                        pix = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5),
                                               alpha=False)
                        pix.save(image_path)
                    width, height = _thumbnail(image_path, thumb_path)
                    records.append({
                        "image_id": iid, "slide_id": sid,
                        "source_id": source_id, "kind": "full_slide",
                        "image_path": _relative(image_path),
                        "thumb_path": _relative(thumb_path),
                        "width": width, "height": height,
                    })

    index_path = Path(PATHS["images_index"])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    print(f"Extraction complete: {len(records)} images indexed.")
    return len(records)
