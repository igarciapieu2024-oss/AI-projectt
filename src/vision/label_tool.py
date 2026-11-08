"""Page to label slide images by hand in 6 classes.

Run from the repo root:
    streamlit run src/vision/label_tool.py

Shows one image at a time; each click saves one row (image_id, label, labeller) in
data/labels/image_labels.csv. These labels train the CNN (src/train/train_image_classifier.py).
Target: about 300 images, at least 20 per class.
"""
from __future__ import annotations  # lets us write types like "list[dict]" in any Python version

import csv  # reads and writes the labels file
import sys  # lets us add the repo folder to Python's search path
from collections import Counter  # counts how many images there are of each class
from pathlib import Path  # handles file paths the same way on Windows and Mac

# streamlit runs this file directly, not as part of the project, so Python doesn't know where "src" is;
# we add the repo folder to its search path so "from src.config import PATHS" works
ROOT = Path(__file__).resolve().parents[2]  # this file is in src/vision/, so 2 folders up is the repo
if str(ROOT) not in sys.path:  # only if it isn't there already
    sys.path.insert(0, str(ROOT))  # adds the repo folder at the start of the search path

import streamlit as st  # noqa: E402  the library that builds the web page (imported after fixing the path)

from src.config import PATHS  # noqa: E402  the shared folder paths of the project

# the 6 classes, in the same order as in the CNN
CLASSES = ["diagram", "chart", "formula", "table", "photo", "decoration"]
# the rule for each class, shown next to the buttons so everyone labels with the same criteria
HELP = {
    "diagram": "boxes and arrows, schemes, processes",
    "chart": "plots with axes: bars, lines, scatter",
    "formula": "equations shown as an image",
    "table": "a table pasted as an image",
    "photo": "real photos: people, buildings, products",
    "decoration": "icons, backgrounds, ornaments with no information",
}
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}  # file types we show
LABELS_FILE = PATHS["image_labels"]  # data/labels/image_labels.csv (this one IS uploaded to GitHub)
FIELDS = ["image_id", "label", "labeller"]  # the 3 columns of the labels file


#1 read and write the labels file

def load_labels() -> list[dict]:
    """All the labels saved so far (empty list the first time)."""
    if not LABELS_FILE.exists():  # nobody has labelled anything yet
        return []  # so there are no labels
    with LABELS_FILE.open(encoding="utf-8", newline="") as f:  # opens the file
        return list(csv.DictReader(f))  # one dictionary per row: image_id, label, labeller


def save_label(image_id: str, label: str, labeller: str) -> None:
    """Adds one row at the end of the file; it never rewrites the file, so a crash can't lose earlier labels."""
    LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)  # creates data/labels/ if it doesn't exist
    new_file = not LABELS_FILE.exists()  # True the very first time
    with LABELS_FILE.open("a", encoding="utf-8", newline="") as f:  # "a" = append at the end
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:  # only the first time
            writer.writeheader()  # writes the column names
        writer.writerow({"image_id": image_id, "label": label, "labeller": labeller})  # the new label


def all_images() -> list[Path]:
    """The images from the extraction step, in data/interim/images/, sorted so the order is always the same."""
    folder = PATHS["images"]  # data/interim/images/
    if not folder.exists():  # the images haven't been extracted on this laptop yet
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_TYPES)  # only image files


#2 the page

st.set_page_config(page_title="Label images", page_icon="🏷️", layout="centered")  # browser tab title and layout
st.title("Label slide images")  # the page title

# sidebar: who is labelling and how many they want to do
labeller = st.sidebar.text_input("Your name", key="labeller")  # saved with each label, to know who labelled what
target = st.sidebar.number_input("Your target", min_value=10, max_value=1000, value=150, step=10)  # for the progress bar

labels = load_labels()  # every label saved so far
done_ids = {row["image_id"] for row in labels}  # images already labelled, by anyone, so they are not shown again
counts = Counter(row["label"] for row in labels)  # how many images of each class so far
mine = sum(1 for row in labels if row["labeller"] == labeller)  # how many this person has done

# shows the count per class and warns when a class has fewer than 20, because the CNN
# needs enough examples of every class (otherwise it never learns the rare ones, like formulas)
st.sidebar.markdown("**Labels so far (everyone)**")
for c in CLASSES:  # one line per class
    flag = "" if counts[c] >= 20 else "  ← needs 20"  # the warning when a class is short
    st.sidebar.write(f"{c}: {counts[c]}{flag}")
st.sidebar.caption("Classes")
for c in CLASSES:
    st.sidebar.caption(f"**{c}**: {HELP[c]}")  # the rules, always visible while labelling

images = all_images()  # all the extracted images
if not images:  # the image extraction hasn't been run on this laptop yet
    st.warning(f"No images found in {PATHS['images'].relative_to(ROOT)}/. Run the image extraction first.")
    st.stop()  # stops the page here, there is nothing to label
if not labeller:  # we need a name before saving anything
    st.info("Write your name in the sidebar to start.")
    st.stop()

# images skipped with "Skip" are remembered only while the page is open, so they come back next time
skipped = st.session_state.setdefault("skipped", set())
todo = [p for p in images if p.stem not in done_ids and p.stem not in skipped]  # what is left to label
st.progress(min(mine / target, 1.0), text=f"You: {mine} / {target} · images left: {len(todo)}")  # progress bar

if not todo:  # everything is labelled
    st.success("No images left to label.")
    st.stop()

current = todo[0]  # the next image to label
st.image(str(current), caption=current.stem)  # the caption is the image_id, to know which slide it comes from

# one button per class, in 3 columns; clicking saves the label and reloads the page with the next image
cols = st.columns(3)  # 3 columns, so the 6 buttons fit in 2 rows
for i, c in enumerate(CLASSES):  # one button per class
    if cols[i % 3].button(c.capitalize(), key=f"btn_{c}", help=HELP[c]):  # i % 3 places it in column 0, 1 or 2
        save_label(current.stem, c, labeller)  # saves the label (the file name without extension is the image_id)
        st.rerun()  # reloads, so the image we just labelled disappears from the list

# when we are not sure, it's better to skip than to guess: a wrong label teaches the CNN the wrong thing
if st.button("Skip (not sure)", key="skip"):
    skipped.add(current.stem)  # remembers it for this session
    st.rerun()  # shows the next image