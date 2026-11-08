"""Describe the useful slide images with a local vision model in Ollama, once (owner: Irene).

Run from the repo root (Ollama must be open):
    python -m src.vision.describe --limit 5         # try it on 5 images first
    python -m src.vision.describe                  # all the useful images
    python -m src.vision.describe --review 50      # 50 random descriptions to check by hand
    python -m src.vision.describe --apply-review   # saves your corrections
"""
from __future__ import annotations  # lets us write types like "int | None" in any Python version

import argparse  # reads the options we write after the command (--limit, --review...)
import csv  # reads the classes file and the review spreadsheet
import json  # writes the review counts as a readable file
import random  # picks the random sample for the check by hand
import sys  # returns the exit code when the script finishes
import time  # measures how long each description takes
from pathlib import Path  # handles file paths the same way on Windows and Mac

from src.config import PATHS, load_config  # the shared folder paths and settings of the project
from src.schemas import ImageText  # the agreed format of one image row
from src.utils.io import read_jsonl, write_jsonl  # read and write .jsonl files (one row per line)

# only these 4 types carry information; photos and decorations are skipped because describing
# each image takes several seconds and they would add nothing a student could search for
USEFUL_CLASSES = {"diagram", "chart", "formula", "table"}
MAX_WORDS = 120  # short enough to fit inside the slide's chunk without pushing out the slide text
REVIEW_FILE = PATHS["labels"] / "description_review.csv"  # the spreadsheet for the check by hand
# the columns of that spreadsheet: the first 5 are filled by the script, the last 3 by us
REVIEW_FIELDS = ["image_id", "image_class", "image_path", "ocr_text", "description", "correct", "fixed_description", "error_type"]

# the instruction we give the vision model. Each line has a reason:
# - "only what is visible": small vision models tend to add general knowledge that is not in the image
# - the class: tells it what to look for (axes in a chart, boxes and arrows in a diagram)
# - labels in order: those words are what students will type when they search
# - the OCR text: the vision model often misreads small words, the OCR text helps it spell them right
PROMPT = """You are describing one image from a university course slide, so that students can find it by searching.
Describe only what is visible in the image. Do not add knowledge that is not in the image.
1. Say what kind of image it is ({image_class}).
2. List the labels, arrows, axes, values and steps you can read, in their order.
3. Say in one sentence what the image shows.
Use at most {max_words} words.
Text read from the image (it may have errors; use it to spell words correctly): "{ocr_text}"
"""


#1 describe one image

def describe_image(path: str, ocr_text: str, cfg, image_class: str = "diagram") -> str:
    """Send one image + the prompt to the vision model and return its description."""
    import ollama  # imported here so the rest of the project works even if Ollama is not installed

    # fills the prompt; the OCR text is cut at 1,500 characters so a very long one doesn't flood the model
    prompt = PROMPT.format(image_class=image_class, max_words=MAX_WORDS, ocr_text=(ocr_text or "")[:1500])
    reply = ollama.chat(  # sends the request to Ollama, which runs on this laptop
        model=cfg.generation["vision_model"],  # moondream by default, set in configs/base.yaml (not written here)
        messages=[{"role": "user", "content": prompt, "images": [str(path)]}],  # the image goes with the message
        options={"temperature": 0},  # 0 = no randomness, the same image always gets the same description
    )
    # newer versions of the ollama library return an object, older ones a dictionary; this works with both
    text = reply["message"]["content"] if isinstance(reply, dict) else reply.message.content
    words = " ".join(text.split()).split(" ")  # removes line breaks and double spaces, then splits into words
    return " ".join(words[:MAX_WORDS])  # cuts it at 120 words in case the model wrote more


#2 read what we need: the image rows from the extraction step and the CNN classes

def load_classes() -> dict[str, tuple[str, float]]:
    """image_id -> (class, probability), written by the CNN (train_image_classifier.py)."""
    path = PATHS["image_classes"]  # data/interim/image_classes.csv
    if not path.exists():  # the CNN hasn't classified the images yet
        raise SystemExit(f"No image classes yet: {path}. Run: python -m src.train.train_image_classifier")
    with path.open(encoding="utf-8", newline="") as f:  # opens the file
        # one entry per image: its class and how sure the CNN was
        return {r["image_id"]: (r["image_class"], float(r["class_prob"])) for r in csv.DictReader(f)}


def load_rows() -> list[ImageText]:
    """One row per image from data/image_texts.jsonl: its ID, its slide, its file and its OCR text."""
    path = PATHS["image_texts"]  # data/image_texts.jsonl
    if not path.exists():  # the extraction step hasn't been run yet
        raise SystemExit(f"No {path.name} yet: it is written by the image extraction + OCR step.")
    # ImageText (src/schemas.py) checks every row has exactly the agreed fields, so a typo fails here, not later
    return read_jsonl(path, ImageText)  # type: ignore[return-value]


#3 describe all the useful images, saving as we go

def describe_all(cfg, limit: int | None = None, force: bool = False) -> int:
    """Add the class to every row and a description to the useful ones. Returns how many were described."""
    rows = load_rows()  # all the image rows
    classes = load_classes()  # the CNN class of each image
    model = cfg.generation["vision_model"]  # the vision model name, from the settings

    for row in rows:  # copies the CNN class into each row, so the chunks know what each image is
        if row.image_id in classes:  # only images the CNN classified
            row.image_class, row.class_prob = classes[row.image_id]

    # what still needs a description: useful class, no description yet (unless --force), and not corrected by hand
    # (we never overwrite a description we fixed by hand)
    todo = [r for r in rows if r.image_class in USEFUL_CLASSES and (force or not r.description) and not r.human_checked]
    if limit:  # --limit 5 to try it quickly before running everything
        todo = todo[:limit]
    # prints how many images there are in total, how many are useful and how many are left to describe
    print(f"{len(rows)} images · {sum(r.image_class in USEFUL_CLASSES for r in rows)} useful · {len(todo)} to describe with {model}")

    done, start = 0, time.perf_counter()  # a counter and the start time, to measure the speed
    for row in todo:  # one image at a time
        path = Path(row.image_path)  # the image file
        if not path.is_absolute():  # paths in the file are relative to the repo, so they work on every laptop
            path = PATHS["root"] / path  # we add the repo folder in front
        if not path.exists():  # the file is listed but not on this laptop
            print(f"  missing image file: {row.image_path}")
            continue  # skip it and go to the next one
        try:  # we catch errors so one bad image doesn't stop the whole run
            row.description = describe_image(str(path), row.ocr_text, cfg, row.image_class)  # step 1
            row.vision_model = model  # we save which model wrote it, for the documentation
        except Exception as exc:  # noqa: BLE001
            # if Ollama fails on one image we don't stop everything: that row stays empty and is retried next run
            print(f"  {row.image_id}: {type(exc).__name__}: {exc}")
            continue
        done += 1  # one more described
        elapsed = time.perf_counter() - start  # seconds since we started
        print(f"  {done}/{len(todo)} {row.image_id} ({elapsed / done:.1f} s/image)")  # shows the speed, to estimate the total time
        if done % 10 == 0:  # every 10 images
            # saves the file: if the laptop sleeps or Ollama closes, we lose at most 10 and the next run continues here
            write_jsonl(PATHS["image_texts"], rows)
    write_jsonl(PATHS["image_texts"], rows)  # final save
    print(f"Described {done} images -> {PATHS['image_texts'].relative_to(PATHS['root'])}")
    return done  # how many were described in this run


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
    parser.add_argument("--limit", type=int, help="describe at most this many images (to try it first)")
    parser.add_argument("--force", action="store_true", help="describe again images that already have a description")
    parser.add_argument("--review", type=int, metavar="N", help="write N random descriptions to check by hand")
    parser.add_argument("--apply-review", action="store_true", help="apply the corrections from the review file")
    args = parser.parse_args(argv)
    if args.review:  # --review 50
        write_review(args.review)  # step 4, first half
    elif args.apply_review:  # --apply-review
        apply_review()  # step 4, second half
    else:  # no option: describe the images
        describe_all(load_config(args.config), limit=args.limit, force=args.force)  # steps 1 to 3
    return 0  # 0 = finished without errors


if __name__ == "__main__":  # only runs when the file is executed, not when another file imports it
    sys.exit(main())