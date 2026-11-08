<<<<<<< HEAD
"""Settings and paths (owner: Francesca; base.yaml is shared - tell the group before changing it).

Every system variant is one YAML file in configs/ that starts with `extends: base.yaml`
and changes only what differs. Load it with:

    from src.config import load_config, PATHS
    cfg = load_config("configs/final.yaml")
    cfg.retrieval["top_k"], cfg.kb_sig

`kb_sig` is a short hash of the `kb` group: configs that change only retrieval or
generation share one index; a config that changes the knowledge base gets its own
ChromaDB collection `kb_<kb_sig>` and its own chunk files.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent

PATHS = {
    "root": ROOT,
    "configs": ROOT / "configs",
    "raw_syllabi": ROOT / "data" / "raw" / "syllabi",
    "raw_slides": ROOT / "data" / "raw" / "slides",
    "manifest": ROOT / "data" / "manifest.csv",
    "labels": ROOT / "data" / "labels",
    "image_labels": ROOT / "data" / "labels" / "image_labels.csv",
    "router_questions": ROOT / "data" / "labels" / "router_questions.csv",
    "image_texts": ROOT / "data" / "image_texts.jsonl",
    "interim": ROOT / "data" / "interim",
    "syllabus_sections": ROOT / "data" / "interim" / "syllabus_sections.jsonl",
    "slides": ROOT / "data" / "interim" / "slides.jsonl",
    "images": ROOT / "data" / "interim" / "images",
    "thumbs": ROOT / "data" / "interim" / "thumbs",
    "images_index": ROOT / "data" / "interim" / "images_index.csv",
    "image_classes": ROOT / "data" / "interim" / "image_classes.csv",
    "processed": ROOT / "data" / "processed",
    "course_table": ROOT / "data" / "processed" / "course_table.csv",
    "chroma": ROOT / "chroma_db",
    "models": ROOT / "models",
    "router_model": ROOT / "models" / "router.joblib",
    "router_meta": ROOT / "models" / "router_meta.json",
    "image_clf": ROOT / "models" / "image_clf",
    "questions": ROOT / "eval" / "questions.csv",
    "runs": ROOT / "eval" / "runs",
    "reports": ROOT / "reports",
    "figures": ROOT / "docs" / "figures",
}


def chunks_path(kb_sig: str) -> Path:
    return PATHS["processed"] / f"chunks__{kb_sig}.jsonl"


def parents_path(kb_sig: str) -> Path:
    return PATHS["processed"] / f"parents__{kb_sig}.jsonl"


def bm25_path(kb_sig: str) -> Path:
    return PATHS["processed"] / f"bm25__{kb_sig}.pkl"


GROUPS = ("kb", "retrieval", "routing", "generation", "eval")


@dataclass
class Config:
    name: str
    path: Path
    kb: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    routing: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    eval: dict[str, Any] = field(default_factory=dict)

    @property
    def kb_sig(self) -> str:
        blob = json.dumps(self.kb, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]

    @property
    def collection_name(self) -> str:
        return f"kb_{self.kb_sig}"


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_raw(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    if parent:
        data = _merge(_load_raw(path.parent / parent), data)
    return data


def load_config(path: str | Path = "configs/base.yaml") -> Config:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    data = _load_raw(path)
    unknown = set(data) - set(GROUPS)
    if unknown:
        raise ValueError(f"{path.name}: unknown config groups {sorted(unknown)}")
    return Config(name=path.stem, path=path, **{g: data.get(g, {}) for g in GROUPS})
=======
from pathlib import Path
# ============================================================
# PROJECT DIRECTORIES
# ============================================================

# Automatically identify the project's root directory.
ROOT_DIR = Path(__file__).resolve().parents[1]

# Original course materials.
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"

# Processed data and extracted images.
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
IMAGE_OUTPUT_DIR = DATA_PROCESSED_DIR / "images"

# Local ChromaDB storage.
CHROMA_DB_DIR = ROOT_DIR / "chroma_db"


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

# Supported document formats.
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".pptx",
    ".docx",
}

# Default OCR languages (English and Spanish).
OCR_LANGUAGE = "eng+spa"


# ============================================================
# TEXT CHUNKING
# ============================================================

CHUNK_SIZE = 350
CHUNK_OVERLAP = 50


# ============================================================
# RETRIEVAL
# ============================================================

# Number of relevant chunks retrieved per question.
TOP_K = 5


# ============================================================
# VECTOR DATABASE
# ============================================================

CHROMA_COLLECTION_NAME = "university_courses"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# These will be configured when the team selects its
# local embedding model and LLM.

EMBEDDING_MODEL_NAME = None
LLM_MODEL_NAME = None


# ============================================================
# COURSE DISCOVERY
# ============================================================

def get_course_dirs() -> list[Path]:
    """
    Automatically discover course folders inside data/raw.

    This allows new courses to be added without changing
    the ingestion code.
    """

    if not DATA_RAW_DIR.exists():
        return []

    return sorted(
        folder
        for folder in DATA_RAW_DIR.iterdir()
        if folder.is_dir()
    )


def get_course_names() -> list[str]:
    """Return the names of all discovered courses."""

    return [
        folder.name
        for folder in get_course_dirs()
    ]
>>>>>>> 0eadc24 (Implement document extraction, OCR, cleaning and chunking)
