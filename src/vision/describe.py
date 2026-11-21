# Classify and describe every slide image with a local vision model (qwen2.5vl:7b in Ollama), in one step.
# The vision model decides the type of each image and writes its description in the same call, so no
# separate image classifier is needed. So we have one model, one decision, and less noise in the knowledge base.
# This file reads data/image_texts.jsonl (written by run_vision.py, one row per image with its OCR text),
# adds a class and a description to each row and saves the file again. chunking.py then adds the OCR
# text and the description of the useful images to the chunk of their slide, and skips photos and
# decorations.
#
# How to run it from the repo folder (Ollama must be open):
#   python -m src.vision.describe --limit 5                        try it on 5 images first
#   python -m src.vision.describe --limit 5 --model qwen2.5vl:3b   try another vision model without editing the config
#   python -m src.vision.describe                                  all the pending images
#   python -m src.vision.describe --review 50                      write 50 random descriptions to check by hand
#   python -m src.vision.describe --apply-review                   save the corrections and count the results
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

from src.config import PATHS, load_config
from src.schemas import ImageText
from src.utils.io import read_jsonl, write_jsonl

# ----------------------------------------------------------------------
# Settings

# The 6 classes the vision model can choose from. They are the same names used in src/schemas.py,
# so every file of the project understands the class written here.
CLASSES = ["diagram", "chart", "formula", "table", "photo", "decoration"]
# Only these 4 classes carry course content. Photos and decorations (logos, icons, backgrounds) get a
# class but no description, and chunking.py leaves them out of the knowledge base.
USEFUL_CLASSES = {"diagram", "chart", "formula", "table"}
# Maximum length of a description. It is added to the chunk of its slide, so it has to be short enough
# not to push out the text of the slide itself (a chunk holds at most 800 characters).
MAX_WORDS = 120
# The vision model reads the image at its own resolution, so a large image takes minutes. Shrinking it to
# 768 pixels on its longest side keeps the labels readable and makes each image much faster.
MAX_SIDE = 768
# Spreadsheet used for the check by hand (data/labels/description_review.csv). The first 5 columns are
# filled by the script; the last 3 are filled by the person who checks the images.
REVIEW_FILE = PATHS["labels"] / "description_review.csv"
REVIEW_FIELDS = ["image_id", "image_class", "image_path", "ocr_text", "description", "correct", "fixed_description", "error_type"]

# The instruction sent to the vision model with every image. It is the text the
# model receives. {max_words} and {ocr_text} are filled in for each image, and the double braces {{ }}
# are written so that Python leaves single braces in the final text. Each part has a reason:
#   - the classes are listed with examples, because logos and university branding are the main noise;
#   - photos and decorations get an empty description, so the model itself filters what is not worth storing;
#   - "describe only what is visible", because small vision models tend to add knowledge not in the image;
#   - labels in their order, because those are the words students type when they search;
#   - the OCR text is included, because the vision model often misreads small words;
#   - JSON only, so the program can read the class and the description without guessing.
PROMPT = """You are processing one image taken from a university course slide.

Step 1. Classify the image into exactly one of these classes:
- diagram: boxes, arrows, flows, networks, architectures, process steps
- chart: plots, graphs, bar/line/pie charts, anything with axes or data
- formula: mathematical equations or notation
- table: rows and columns of values or text
- photo: a picture of people, places or objects that teaches nothing about the course content
- decoration: logos, icons, university branding, backgrounds, clip art, stock images

Step 2. If the class is diagram, chart, formula or table, describe it so that students can find it by searching:
- describe only what is visible; do not add knowledge that is not in the image
- list the labels, arrows, axes, values and steps you can read, in their order
- end with one sentence saying what the image shows
- use at most {max_words} words
If the class is photo or decoration, the description must be an empty string.

Text read from the image by OCR (it may have errors; use it to spell words correctly): "{ocr_text}"

Answer with JSON only, in this exact format:
{{"image_class": "<one of the 6 classes>", "description": "<text or empty>"}}
"""


# ----------------------------------------------------------------------
# Classify and describe one image

# Return a smaller copy of the image as PNG bytes, ready to send to the model. The copy is made in
# memory, so the image file on disk is never changed.
def shrink(path: str) -> bytes:
    # Imported here because only this function needs them.
    from io import BytesIO
    from PIL import Image

    with Image.open(path) as im:
        # Many formulas are black text on a transparent background. Converting them straight to RGB turns
        # the transparent parts black and hides the text, so the transparent parts are painted white first.
        if im.mode in ("RGBA", "LA") or "transparency" in im.info:
            im = im.convert("RGBA")
            white = Image.new("RGB", im.size, "white")
            white.paste(im, mask=im.split()[-1])  # the last channel (alpha) says which pixels are visible
            im = white
        # The same colour format for every image.
        im = im.convert("RGB")
        # Shrink keeping the proportions; images already smaller than MAX_SIDE are not enlarged.
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        # Save the copy into a buffer in memory instead of a file.
        buffer = BytesIO()
        im.save(buffer, format="PNG")
    return buffer.getvalue()


# Send one image with the prompt to the vision model and return (class, description).
# Called by describe_all() for every pending image.
def classify_and_describe(path: str, ocr_text: str, model: str) -> tuple[str, str]:
    # Imported here so the rest of the project works on a laptop without the ollama library.
    import ollama

    # Fill the prompt for this image. The OCR text is cut at 1,500 characters, so a very long one
    # does not fill the prompt and push out the instructions.
    prompt = PROMPT.format(max_words=MAX_WORDS, ocr_text=(ocr_text or "")[:1500])
    # The request to Ollama, which runs the model on this laptop.
    reply = ollama.chat(
        model=model,
        # The prompt and the smaller copy of the image go in the same message.
        messages=[{"role": "user", "content": prompt, "images": [shrink(path)]}],
        # Forces the model to answer with JSON, so the class and the description can always be read.
        format="json",
        # temperature 0: no randomness, the same image always gets the same answer.
        # num_predict 400: stops the answer at about 400 tokens; 120 words fit easily.
        # repeat_penalty 1.15: stops the model from listing the same label again and again.
        options={"temperature": 0, "num_predict": 400, "repeat_penalty": 1.15},
        # Keeps the model loaded for 30 minutes, so it is not reloaded for every image.
        keep_alive="30m",
    )
    # Newer versions of the ollama library return an object and older ones a dictionary; this reads both.
    text = reply["message"]["content"] if isinstance(reply, dict) else reply.message.content
    try:
        # Turn the JSON text into a Python dictionary.
        data = json.loads(text)
    except json.JSONDecodeError:
        # The answer was cut before the JSON closed (the 400-token limit was reached). Instead of losing
        # the image, the class and the description are recovered from the text that did arrive.
        import re
        found_class = re.search(r'"image_class"\s*:\s*"([a-zA-Z]+)', text)
        found_desc = re.search(r'"description"\s*:\s*"(.*)', text, re.S)
        data = {"image_class": found_class.group(1) if found_class else "",
                "description": found_desc.group(1).rstrip('"} \n') if found_desc else ""}

    # Clean the class: lower case, and only the class name when the model wrote more.
    image_class = str(data.get("image_class", "")).strip().lower()
    image_class = next((c for c in CLASSES if image_class.startswith(c)), image_class)
    # A class that is not one of the 6 raises an error: the row stays unclassified and is retried
    # in the next run, instead of saving a wrong class.
    if image_class not in CLASSES:
        raise ValueError(f"unknown class from the model: {image_class!r}")
    # Photos and decorations get no description, even if the model wrote one. This is what keeps
    # logos and pictures out of the knowledge base.
    if image_class not in USEFUL_CLASSES:
        return image_class, ""

    # Remove line breaks and double spaces, then cut at MAX_WORDS in case the model wrote more.
    words = " ".join(str(data.get("description", "")).split()).split(" ")
    return image_class, " ".join(words[:MAX_WORDS]).strip()


# ----------------------------------------------------------------------
# Read the image rows and decide which ones are pending

# Read data/image_texts.jsonl: one row per image with its id, slide, file and OCR text.
# ImageText (src/schemas.py) checks that every row has the agreed fields, so a wrong field fails here
# and not later in the chunking.
def load_rows() -> list[ImageText]:
    path = PATHS["image_texts"]
    # The file is written by run_vision.py; without it there is nothing to describe.
    if not path.exists():
        raise SystemExit(f"No {path.name} yet: it is written by the image extraction + OCR step.")
    return read_jsonl(path, ImageText)  # type: ignore[return-value]


# True if the image still has to go through the vision model. This is what lets the run be stopped
# and continued later without repeating the images already done.
def needs_work(row: ImageText, force: bool) -> bool:
    # Rows corrected by hand are never sent again, not even with --force, so a correction is never lost.
    if row.human_checked:
        return False
    # --force: redo every image that was not corrected by hand.
    if force:
        return True
    # Never classified.
    if not row.image_class:
        return True
    # Classified as useful but without a description, for example because Ollama stopped half way.
    return row.image_class in USEFUL_CLASSES and not row.description


# ----------------------------------------------------------------------
# Classify and describe all the pending images

# Add a class to every pending image and a description to the useful ones, saving the file as it goes.
# Returns how many images were processed. Called by main().
def describe_all(cfg, limit: int | None = None, force: bool = False, model: str | None = None) -> int:
    rows = load_rows()
    # The vision model comes from the config (generation.vision_model in base.yaml); --model replaces it,
    # to compare models without editing the YAML.
    model = model or cfg.generation["vision_model"]

    # The images still to process; --limit keeps only the first ones, to try the step quickly.
    todo = [r for r in rows if needs_work(r, force)]
    if limit:
        todo = todo[:limit]
    print(f"{len(rows)} images · {len(todo)} to classify and describe with {model}")

    # Counter and start time, to show the speed and estimate how long the whole run will take.
    done, start = 0, time.perf_counter()
    for row in todo:
        # Paths are stored relative to the repo so they work on every laptop; the repo folder is added in front.
        path = Path(row.image_path)
        if not path.is_absolute():
            path = PATHS["root"] / path
        # The image is listed but not on this laptop: skip it.
        if not path.exists():
            print(f"  missing image file: {row.image_path}")
            continue
        # Errors are caught image by image, so one bad image does not stop the whole run.
        try:
            row.image_class, row.description = classify_and_describe(str(path), row.ocr_text, model)
            # The vision model gives no probability; 1.0 marks "decided by the vision model".
            row.class_prob = 1.0
            # Which model decided, so the documentation can say it.
            row.vision_model = model
        except Exception as exc:  # noqa: BLE001
            # Bad JSON, unknown class or Ollama error: the row stays empty and is retried in the next run.
            print(f"  {row.image_id}: {type(exc).__name__}: {exc}")
            continue
        done += 1
        elapsed = time.perf_counter() - start
        print(f"  {done}/{len(todo)} {row.image_id} -> {row.image_class} ({elapsed / done:.1f} s/image)")
        # Save every 10 images: if the laptop sleeps or Ollama closes, at most 10 images are lost,
        # and the next run continues from there.
        if done % 10 == 0:
            write_jsonl(PATHS["image_texts"], rows)
    # Final save, with all the rows (processed or not), so no row is lost.
    write_jsonl(PATHS["image_texts"], rows)

    # Number of images in each class, the figures used in the documentation.
    counts = {c: sum(r.image_class == c for r in rows) for c in CLASSES}
    print(f"Processed {done} images -> {PATHS['image_texts'].relative_to(PATHS['root'])}")
    print("Images per class:", counts)
    return done


# ----------------------------------------------------------------------
# Check by hand: random descriptions, corrected and counted

# First half of the check: write n random descriptions to a spreadsheet, so a person can compare
# each one with its image.
def write_review(n: int = 50) -> Path:
    # Only images with a description can be checked.
    rows = [r for r in load_rows() if r.description]
    # Shuffled so the sample is random and fair; the fixed seed (42) gives the same sample if it is rerun.
    random.Random(42).shuffle(rows)
    REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        # The first n rows after shuffling are the n random images.
        for r in rows[:n]:
            # The last 3 columns are left empty for the reviewer.
            writer.writerow({"image_id": r.image_id, "image_class": r.image_class, "image_path": r.image_path,
                             "ocr_text": r.ocr_text, "description": r.description,
                             "correct": "", "fixed_description": "", "error_type": ""})
    # What to write in each empty column.
    print(f"Open {REVIEW_FILE.relative_to(PATHS['root'])}, look at each image, and fill in:")
    print("  correct = yes / partly / no · fixed_description = your corrected text (if not yes)")
    print("  error_type = missing text / wrong text / invented content / too vague")
    return REVIEW_FILE


# Second half of the check: save the corrections into image_texts.jsonl and count the results in
# reports/description_review.json. Those counts are the figures of the documentation
# (39 correct, 10 partly correct, 1 wrong out of 50).
def apply_review() -> dict:
    if not REVIEW_FILE.exists():
        raise SystemExit("No review file: run with --review 50 first")
    # Only the rows the reviewer filled in (the "correct" column is not empty).
    with REVIEW_FILE.open(encoding="utf-8", newline="") as f:
        review = {r["image_id"]: r for r in csv.DictReader(f) if r["correct"].strip()}
    rows = load_rows()
    for row in rows:
        r = review.get(row.image_id)
        # Images outside the sample are left as they are.
        if r is None:
            continue
        # A corrected description replaces the model's one.
        if r["fixed_description"].strip():
            row.description = r["fixed_description"].strip()
        # Marked as checked, so needs_work() never sends it to the model again.
        row.human_checked = True
    write_jsonl(PATHS["image_texts"], rows)

    # Count the verdicts (yes / partly / no) and how many times each type of error appeared.
    verdicts = [r["correct"].strip().lower() for r in review.values()]
    errors: dict[str, int] = {}
    for r in review.values():
        if r["error_type"].strip():
            errors[r["error_type"].strip()] = errors.get(r["error_type"].strip(), 0) + 1
    summary = {"checked": len(verdicts), "correct": verdicts.count("yes"), "partly": verdicts.count("partly"),
               "wrong": verdicts.count("no"), "error_types": errors}
    # Save the counts in reports/ and print them.
    out = PATHS["reports"] / "description_review.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


# ----------------------------------------------------------------------
# Command line

# Read the options written after the command and run the matching part:
# --review N writes the spreadsheet, --apply-review saves the corrections, and with no option
# the pending images are classified and described.
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # The config holds the vision model (generation.vision_model in base.yaml).
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--limit", type=int, help="process at most this many images (to try it first)")
    parser.add_argument("--model", help="vision model to use instead of the one in the config (to compare models)")
    parser.add_argument("--force", action="store_true", help="process again images that were already done (except the ones corrected by hand)")
    parser.add_argument("--review", type=int, metavar="N", help="write N random descriptions to check by hand")
    parser.add_argument("--apply-review", action="store_true", help="apply the corrections from the review file")
    args = parser.parse_args(argv)
    if args.review:
        write_review(args.review)
    elif args.apply_review:
        apply_review()
    else:
        describe_all(load_config(args.config), limit=args.limit, force=args.force, model=args.model)
    # 0 tells the terminal the program finished without errors.
    return 0


# Runs only when the file is executed directly, not when another file imports it.
if __name__ == "__main__":
    sys.exit(main())
