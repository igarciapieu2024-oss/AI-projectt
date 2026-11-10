"""Classify and describe every slide image with a local vision model in Ollama, in one step (owner: Irene).

The vision model decides the type of each image AND writes its description in the same call,
so we don't need a separate image classifier: one model, one decision, less noise in the knowledge base.

Run from the repo root (Ollama must be open):
    python -m src.vision.describe --limit 5                       # try it on 5 images first
    python -m src.vision.describe --limit 5 --model qwen2.5vl:3b   # try another vision model without touching the config
    python -m src.vision.describe                                 # all the images
    python -m src.vision.describe --review 50                     # 50 random descriptions to check by hand
    python -m src.vision.describe --apply-review                  # saves your corrections
"""
from __future__ import annotations  # lets us write types like "int | None" in any Python version

import argparse  # reads the options we write after the command (--limit, --review...)
import csv  # reads and writes the review spreadsheet
import json  # reads the model's answer (it comes as JSON) and writes the review counts
import random  # picks the random sample for the check by hand
import sys  # returns the exit code when the script finishes
import time  # measures how long each image takes
from pathlib import Path  # handles file paths the same way on Windows and Mac

from src.config import PATHS, load_config  # the shared folder paths and settings of the project
from src.schemas import ImageText  # the agreed format of one image row
from src.utils.io import read_jsonl, write_jsonl  # read and write .jsonl files (one row per line)

# the 6 types the vision model can choose from (the same 6 names as in src/schemas.py)
CLASSES = ["diagram", "chart", "formula", "table", "photo", "decoration"]
# only these 4 types carry course content; photos and decorations (logos, icons, backgrounds)
# get a class but no description, so they never reach the knowledge base
USEFUL_CLASSES = {"diagram", "chart", "formula", "table"}
MAX_WORDS = 120  # short enough to fit inside the slide's chunk without pushing out the slide text
REVIEW_FILE = PATHS["labels"] / "description_review.csv"  # the spreadsheet for the check by hand
# the columns of that spreadsheet: the first 5 are filled by the script, the last 3 by us
REVIEW_FIELDS = ["image_id", "image_class", "image_path", "ocr_text", "description", "correct", "fixed_description", "error_type"]

# the instruction we give the vision model. Each part has a reason:
# - the list of classes with examples: logos and university branding are the main noise, so we name them
# - "description empty" for photo/decoration: the model itself filters what is not worth storing
# - "only what is visible": small vision models tend to add general knowledge that is not in the image
# - labels in order: those words are what students will type when they search
# - the OCR text: the vision model often misreads small words, the OCR text helps it spell them right
# - JSON only: so the script can read the class and the description without guessing
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


#1 classify and describe one image

def classify_and_describe(path: str, ocr_text: str, model: str) -> tuple[str, str]:
    """Send one image + the prompt to the vision model and return (class, description)."""
    import ollama  # imported here so the rest of the project works even if Ollama is not installed

    # fills the prompt; the OCR text is cut at 1,500 characters so a very long one doesn't flood the model
    prompt = PROMPT.format(max_words=MAX_WORDS, ocr_text=(ocr_text or "")[:1500])
    reply = ollama.chat(  # sends the request to Ollama, which runs on this laptop
        model=model,  # the vision model (from configs/base.yaml, or --model to try another one)
        messages=[{"role": "user", "content": prompt, "images": [str(path)]}],  # the image goes with the message
        format="json",  # forces the model to answer with valid JSON, so we can always read it
        options={"temperature": 0},  # 0 = no randomness, the same image always gets the same answer
    )
    # newer versions of the ollama library return an object, older ones a dictionary; this works with both
    text = reply["message"]["content"] if isinstance(reply, dict) else reply.message.content
    data = json.loads(text)  # turns the JSON text into a Python dictionary

    image_class = str(data.get("image_class", "")).strip().lower()  # the class, in lower case to compare it
    if image_class not in CLASSES:  # the model wrote something that is not one of the 6 classes
        raise ValueError(f"unknown class from the model: {image_class!r}")  # the image is retried in the next run
    if image_class not in USEFUL_CLASSES:  # photo or decoration
        return image_class, ""  # no description: this is what keeps logos and pictures out of the knowledge base

    words = " ".join(str(data.get("description", "")).split()).split(" ")  # removes line breaks and double spaces, then splits into words
    return image_class, " ".join(words[:MAX_WORDS]).strip()  # cuts it at 120 words in case the model wrote more


#2 read the image rows written by the extraction + OCR step

def load_rows() -> list[ImageText]:
    """One row per image from data/image_texts.jsonl: its ID, its slide, its file and its OCR text."""
    path = PATHS["image_texts"]  # data/image_texts.jsonl
    if not path.exists():  # the extraction step hasn't been run yet
        raise SystemExit(f"No {path.name} yet: it is written by the image extraction + OCR step.")
    # ImageText (src/schemas.py) checks every row has exactly the agreed fields, so a typo fails here, not later
    return read_jsonl(path, ImageText)  # type: ignore[return-value]


def needs_work(row: ImageText, force: bool) -> bool:
    """True if this image still has to go through the vision model."""
    if row.human_checked:  # we never overwrite an image we corrected by hand
        return False
    if force:  # --force: redo everything that was not corrected by hand
        return True
    if not row.image_class:  # never classified
        return True
    # classified as useful but the description is missing (for example Ollama failed half way)
    return row.image_class in USEFUL_CLASSES and not row.description


#3 classify and describe all the images, saving as we go

def describe_all(cfg, limit: int | None = None, force: bool = False, model: str | None = None) -> int:
    """Add a class to every image and a description to the useful ones. Returns how many were processed."""
    rows = load_rows()  # all the image rows
    model = model or cfg.generation["vision_model"]  # --model if given, otherwise the one in the settings

    todo = [r for r in rows if needs_work(r, force)]  # what is still pending
    if limit:  # --limit 5 to try it quickly before running everything
        todo = todo[:limit]
    print(f"{len(rows)} images · {len(todo)} to classify and describe with {model}")

    done, start = 0, time.perf_counter()  # a counter and the start time, to measure the speed
    for row in todo:  # one image at a time
        path = Path(row.image_path)  # the image file
        if not path.is_absolute():  # paths in the file are relative to the repo, so they work on every laptop
            path = PATHS["root"] / path  # we add the repo folder in front
        if not path.exists():  # the file is listed but not on this laptop
            print(f"  missing image file: {row.image_path}")
            continue  # skip it and go to the next one
        try:  # we catch errors so one bad image doesn't stop the whole run
            row.image_class, row.description = classify_and_describe(str(path), row.ocr_text, model)  # step 1
            row.class_prob = 1.0  # the vision model gives no probability; 1.0 marks "decided by the vision model"
            row.vision_model = model  # we save which model decided, for the documentation
        except Exception as exc:  # noqa: BLE001
            # bad JSON, unknown class or Ollama error: that row stays empty and is retried next run
            print(f"  {row.image_id}: {type(exc).__name__}: {exc}")
            continue
        done += 1  # one more processed
        elapsed = time.perf_counter() - start  # seconds since we started
        # shows the class it chose and the speed, to estimate the total time
        print(f"  {done}/{len(todo)} {row.image_id} -> {row.image_class} ({elapsed / done:.1f} s/image)")
        if done % 10 == 0:  # every 10 images
            # saves the file: if the laptop sleeps or Ollama closes, we lose at most 10 and the next run continues here
            write_jsonl(PATHS["image_texts"], rows)
    write_jsonl(PATHS["image_texts"], rows)  # final save

    # summary for the documentation: how many images ended up in each class
    counts = {c: sum(r.image_class == c for r in rows) for c in CLASSES}
    print(f"Processed {done} images -> {PATHS['image_texts'].relative_to(PATHS['root'])}")
    print("Images per class:", counts)
    return done  # how many were processed in this run


#4 the check by hand: 50 random descriptions, corrected and counted

def write_review(n: int = 50) -> Path:
    """Pick n random descriptions and write them in a spreadsheet to check against the images."""
    rows = [r for r in load_rows() if r.description]  # only images that have a description
    random.Random(42).shuffle(rows)  # random, so the sample is fair; fixed seed, so it is the same sample if we rerun it
    REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)  # creates data/labels/ if it doesn't exist
    with REVIEW_FILE.open("w", encoding="utf-8", newline="") as f:  # creates the spreadsheet
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()  # the column names
        for r in rows[:n]:  # the first n after shuffling = n random images
            # the last 3 columns are left empty for us to fill in
            writer.writerow({"image_id": r.image_id, "image_class": r.image_class, "image_path": r.image_path,
                             "ocr_text": r.ocr_text, "description": r.description,
                             "correct": "", "fixed_description": "", "error_type": ""})
    # tells us what to write in each empty column
    print(f"Open {REVIEW_FILE.relative_to(PATHS['root'])}, look at each image, and fill in:")
    print("  correct = yes / partly / no · fixed_description = your corrected text (if not yes)")
    print("  error_type = missing text / wrong text / invented content / too vague")
    return REVIEW_FILE


def apply_review() -> dict:
    """Saves our corrections into image_texts.jsonl and counts the errors for the documentation."""
    if not REVIEW_FILE.exists():  # we haven't created the review spreadsheet yet
        raise SystemExit("No review file: run with --review 50 first")
    with REVIEW_FILE.open(encoding="utf-8", newline="") as f:  # opens the spreadsheet we filled in
        review = {r["image_id"]: r for r in csv.DictReader(f) if r["correct"].strip()}  # only the rows we filled in
    rows = load_rows()  # all the image rows
    for row in rows:  # goes through every image
        r = review.get(row.image_id)  # our review of this image, if we reviewed it
        if r is None:  # not in the sample, nothing to do
            continue
        if r["fixed_description"].strip():  # if we wrote a correction
            row.description = r["fixed_description"].strip()  # our corrected version replaces the model's
        row.human_checked = True  # marks it as checked, so a later run never overwrites it
    write_jsonl(PATHS["image_texts"], rows)  # saves the corrections

    # the counts that go into the documentation: how many were right, partly right or wrong, and why
    verdicts = [r["correct"].strip().lower() for r in review.values()]  # yes / partly / no for each image
    errors: dict[str, int] = {}  # error type -> how many times
    for r in review.values():
        if r["error_type"].strip():  # if we wrote an error type
            errors[r["error_type"].strip()] = errors.get(r["error_type"].strip(), 0) + 1  # counts it
    summary = {"checked": len(verdicts), "correct": verdicts.count("yes"), "partly": verdicts.count("partly"),
               "wrong": verdicts.count("no"), "error_types": errors}
    out = PATHS["reports"] / "description_review.json"  # reports/description_review.json
    out.parent.mkdir(parents=True, exist_ok=True)  # creates reports/ if it doesn't exist
    out.write_text(json.dumps(summary, indent=2))  # saves the counts
    print(json.dumps(summary, indent=2))  # and prints them
    return summary


def main(argv: list[str] | None = None) -> int:
    # reads the options written after the command
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/base.yaml")  # which settings to use (the vision model is there)
    parser.add_argument("--limit", type=int, help="process at most this many images (to try it first)")
    parser.add_argument("--model", help="vision model to use instead of the one in the config (to compare models)")
    parser.add_argument("--force", action="store_true", help="process again images that were already done (except the ones corrected by hand)")
    parser.add_argument("--review", type=int, metavar="N", help="write N random descriptions to check by hand")
    parser.add_argument("--apply-review", action="store_true", help="apply the corrections from the review file")
    args = parser.parse_args(argv)
    if args.review:  # --review 50
        write_review(args.review)  # step 4, first half
    elif args.apply_review:  # --apply-review
        apply_review()  # step 4, second half
    else:  # no option: classify and describe the images
        describe_all(load_config(args.config), limit=args.limit, force=args.force, model=args.model)  # steps 1 to 3
    return 0  # 0 = finished without errors


if __name__ == "__main__":  # only runs when the file is executed, not when another file imports it
    sys.exit(main())