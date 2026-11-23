# Extract the images of the slide PDFs so their content can be read in the next steps.
# Each image is saved once, together with a small thumbnail, and listed in
# data/interim/images_index.csv. run_vision.py reads that list to run OCR on every image,
# and the app uses it to show the thumbnails of full-slide renders next to the sources.
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

# Columns of images_index.csv. "kind" says whether the row is an image embedded in the
# slide or a render of the whole slide.
INDEX_FIELDS = ["image_id", "slide_id", "source_id", "kind",
                "image_path", "thumb_path", "width", "height"]
# Images below this size are icons or bullets and carry no course content.
MIN_WIDTH, MIN_HEIGHT, MIN_PIXELS = 80, 80, 8000
# Maximum size of the thumbnails, small enough to load quickly in the app.
THUMB_SIZE = (480, 360)


# Store paths relative to the repo, so the index works on every laptop.
def _relative(path: Path) -> str:
    return path.relative_to(Path(PATHS["root"])).as_posix()


# Save a JPEG thumbnail of the image and return the original width and height,
# which are measured before the image is shrunk.
def _thumbnail(image_path: Path, thumb_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as im:
        # JPEG has no transparency channel, so every image is converted to RGB first.
        im = im.convert("RGB")
        size = im.size
        # thumbnail() shrinks the image in place and keeps its proportions.
        im.thumbnail(THUMB_SIZE)
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(thumb_path, "JPEG", quality=85, optimize=True)
    return size


# Extract every unique embedded image from the slide PDFs and, for the slides flagged in
# parse_slides.py as having little or garbled text, also render the whole page as an image,
# so the OCR can read content the PDF text extraction missed. Returns the number of images indexed.
def extract_images(cfg=None) -> int:
    # Folders: the PDFs are read from data/raw, the images and thumbnails are written
    # to the folders defined in config.py.
    root = Path(PATHS["root"])
    raw_root = root / "data" / "raw"
    images_dir, thumbs_dir = Path(PATHS["images"]), Path(PATHS["thumbs"])
    images_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    # The manifest lists every PDF with its source_id and document type.
    manifest = read_csv(PATHS["manifest"])
    # The flags come from data/interim/slides.jsonl, written by parse_slides.py.
    # If that file does not exist yet, no slide is rendered whole.
    slides_file = Path(PATHS["slides"])
    slide_rows = read_jsonl(slides_file, SlideRecord) if slides_file.is_file() else []
    flagged = {(s.source_id, s.page) for s in slide_rows
               if s.is_low_text or s.is_garbled}

    # One dictionary per image, written to images_index.csv at the end.
    records = []
    # Hashes of the images already saved. A logo or background repeated on many slides is
    # the same file each time, so comparing hashes keeps only its first occurrence.
    seen_hashes = set()

    for source in manifest:
        # Only slide decks have images to extract; syllabi are skipped.
        if source.get("doc_type") != "slides":
            continue
        source_id = source["source_id"]
        pdf_path = raw_root / source["file"]
        # A PDF listed in the manifest but missing on disk stops the run, so an incomplete
        # extraction is not taken as complete.
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF missing for {source_id}: {pdf_path}")

        with pymupdf.open(pdf_path) as pdf:
            # Pages are numbered from 1, so the slide_id matches the one built in parse_slides.py
            # and each image can be joined to its slide.
            for page_no, page in enumerate(pdf, start=1):
                sid = make_slide_id(source_id, page_no)
                # Counter of the images kept on this slide, used in the image_id.
                image_no = 0

                # Images embedded in the page. info[0] is the internal reference PyMuPDF
                # uses to extract the image bytes.
                for info in page.get_images(full=True):
                    extracted = pdf.extract_image(info[0])
                    data = extracted.get("image", b"")
                    if not data:
                        continue
                    digest = hashlib.sha256(data).hexdigest()
                    if digest in seen_hashes:
                        continue
                    # Skip images that Pillow cannot open, so a damaged image does not stop
                    # the extraction or fail later in the OCR.
                    try:
                        with Image.open(BytesIO(data)) as im:
                            width, height = im.size
                            im.verify()
                    except Exception:
                        continue
                    if (width < MIN_WIDTH or height < MIN_HEIGHT
                            or width * height < MIN_PIXELS):
                        continue

                    # The image is kept: register its hash and give it an id.
                    seen_hashes.add(digest)
                    image_no += 1
                    iid = make_image_id(sid, image_no)
                    # Keep the original format when it is a common one, otherwise save as PNG.
                    ext = extracted.get("ext", "png").lower()
                    if ext == "jpeg":
                        ext = "jpg"
                    if ext not in {"png", "jpg", "webp"}:
                        ext = "png"
                    image_path = images_dir / f"{iid}.{ext}"
                    thumb_path = thumbs_dir / f"{iid}.jpg"
                    # Existing files are not rewritten, so a second run is faster.
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

                # Full-page render for flagged slides. The 1.5 zoom gives a higher resolution
                # than the default, so small text stays readable for the OCR. alpha=False
                # gives a solid background instead of a transparent one.
                if (source_id, page_no) in flagged:
                    # Without a number, image_id marks the render as the whole slide.
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

    # Write the index read by run_vision.py and by the app. It is rewritten on every run,
    # so it always matches the images currently extracted.
    index_path = Path(PATHS["images_index"])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    print(f"Extraction complete: {len(records)} images indexed.")
    return len(records)
