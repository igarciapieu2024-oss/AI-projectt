"""Label slide images in 6 classes -> data/labels/image_labels.csv.

Owner: Irene. Run from the repo root:
    streamlit run src/vision/label_tool.py

It shows one image at a time from data/interim/images/ (made by Santiago's extraction),
skips the ones already labelled, and appends one row per click:
    image_id, label, labeller
Target: at least 20 images per class, about 300 in total (150 Irene, 150 Francesca).
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.config import PATHS  # noqa: E402

CLASSES = ["diagram", "chart", "formula", "table", "photo", "decoration"]
HELP = {
    "diagram": "boxes and arrows, schemes, processes",
    "chart": "plots with axes: bars, lines, scatter",
    "formula": "equations shown as an image",
    "table": "a table pasted as an image",
    "photo": "real photos: people, buildings, products",
    "decoration": "icons, backgrounds, ornaments with no information",
}
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
LABELS_FILE = PATHS["image_labels"]
FIELDS = ["image_id", "label", "labeller"]


def load_labels() -> list[dict]:
    if not LABELS_FILE.exists():
        return []
    with LABELS_FILE.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_label(image_id: str, label: str, labeller: str) -> None:
    LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LABELS_FILE.exists()
    with LABELS_FILE.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({"image_id": image_id, "label": label, "labeller": labeller})


def all_images() -> list[Path]:
    folder = PATHS["images"]
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_TYPES)


st.set_page_config(page_title="Label images", page_icon="🏷️", layout="centered")
st.title("Label slide images")

labeller = st.sidebar.text_input("Your name", key="labeller")
target = st.sidebar.number_input("Your target", min_value=10, max_value=1000, value=150, step=10)

labels = load_labels()
done_ids = {row["image_id"] for row in labels}
counts = Counter(row["label"] for row in labels)
mine = sum(1 for row in labels if row["labeller"] == labeller)

st.sidebar.markdown("**Labels so far (everyone)**")
for c in CLASSES:
    flag = "" if counts[c] >= 20 else "  ← needs 20"
    st.sidebar.write(f"{c}: {counts[c]}{flag}")
st.sidebar.caption("Classes")
for c in CLASSES:
    st.sidebar.caption(f"**{c}**: {HELP[c]}")

images = all_images()
if not images:
    st.warning(f"No images found in {PATHS['images'].relative_to(ROOT)}/. Run Santiago's extraction first.")
    st.stop()
if not labeller:
    st.info("Write your name in the sidebar to start.")
    st.stop()

skipped = st.session_state.setdefault("skipped", set())
todo = [p for p in images if p.stem not in done_ids and p.stem not in skipped]
st.progress(min(mine / target, 1.0), text=f"You: {mine} / {target} · images left: {len(todo)}")

if not todo:
    st.success("No images left to label.")
    st.stop()

current = todo[0]
st.image(str(current), caption=current.stem)

cols = st.columns(3)
for i, c in enumerate(CLASSES):
    if cols[i % 3].button(c.capitalize(), key=f"btn_{c}", help=HELP[c]):
        save_label(current.stem, c, labeller)
        st.rerun()

if st.button("Skip (not sure)", key="skip"):
    skipped.add(current.stem)
    st.rerun()
