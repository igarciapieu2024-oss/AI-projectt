# Settings and file paths shared by the whole project.
# Every version of the system is one YAML file in configs/ that starts with "extends:" and lists only what it
# changes; load_config() reads it and fills in the rest from the file it extends (base.yaml).
# kb_sig is a short code calculated from the kb group of the settings: configurations that only change the
# search or the answers share one knowledge base, and a configuration that changes the knowledge base gets
# its own ChromaDB collection (kb_<kb_sig>) and its own chunk files, so no version overwrites another.
from __future__ import annotations

# hashlib and json are used to calculate kb_sig from the settings.
import hashlib
import json
# dataclass builds the Config class from its list of fields.
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The library that reads YAML files.
import yaml

# The repository folder (two levels above this file), so every path works on every laptop.
ROOT = Path(__file__).resolve().parent.parent

# Every file and folder the project reads or writes, in one place. Some entries (raw_syllabi, raw_slides,
# image_labels, image_classes, image_clf, questions, runs) are not used by the current code: they come from
# the first plan of the project (for example the image classifier that was replaced by the vision model).
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


# The files of one knowledge base, named with its kb_sig. First the chunks.
def chunks_path(kb_sig: str) -> Path:
    return PATHS["processed"] / f"chunks__{kb_sig}.jsonl"


# The parents: the whole section or slide each chunk was cut from (saved, but the answer phase doesn't use them).
def parents_path(kb_sig: str) -> Path:
    return PATHS["processed"] / f"parents__{kb_sig}.jsonl"


# The BM25 index.
def bm25_path(kb_sig: str) -> Path:
    return PATHS["processed"] / f"bm25__{kb_sig}.pkl"


# The five groups of settings a config can have: kb (how the knowledge base is built), retrieval (how it is
# searched), routing (the question classifier), generation (how the answer is written and checked) and
# eval (the judge model).
GROUPS = ("kb", "retrieval", "routing", "generation", "eval")


# One loaded configuration: its name (the file name without .yaml), its path and its five groups.
@dataclass
class Config:
    name: str
    path: Path
    kb: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    routing: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    eval: dict[str, Any] = field(default_factory=dict)

    # The code of the knowledge base: the first 8 characters of a hash of the kb group. The settings are
    # written in a fixed order (sort_keys), so the same settings always give the same code.
    @property
    def kb_sig(self) -> str:
        # Write the kb settings as text, always in the same order.
        blob = json.dumps(self.kb, sort_keys=True, default=str)
        # Hash that text and keep the first 8 characters.
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]

    # The name of the ChromaDB collection of this knowledge base.
    @property
    def collection_name(self) -> str:
        return f"kb_{self.kb_sig}"


# Combine two sets of settings: the values of override replace those of base, group by group, so a config
# only has to list what it changes.
def _merge(base: dict, override: dict) -> dict:
    # Start from a copy of the base settings.
    out = dict(base)
    for key, value in override.items():
        # Inside a group, only the listed settings are replaced; the others keep their base value.
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


# Read a YAML file and, if it starts with "extends:", first read the file it extends (which can extend
# another one) and apply this file's settings on top.
def _load_raw(path: Path) -> dict:
    # Read the YAML file as a dictionary (an empty file gives an empty dictionary).
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # Take "extends" out of the settings; it only says which file comes first.
    parent = data.pop("extends", None)
    if parent:
        data = _merge(_load_raw(path.parent / parent), data)
    return data


# Load a configuration by its path (relative to the repository or absolute). Used by every part of the
# project: the build, answer(), the app and the evaluation.
def load_config(path: str | Path = "configs/base.yaml") -> Config:
    # A relative path is read from the repository folder.
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    # The settings of this file, already merged with the files it extends.
    data = _load_raw(path)
    # A group name that is not one of the five is a typo in the YAML: stop instead of ignoring it.
    unknown = set(data) - set(GROUPS)
    if unknown:
        raise ValueError(f"{path.name}: unknown config groups {sorted(unknown)}")
    # Build the Config: its name is the file name without .yaml, and a missing group is left empty.
    return Config(name=path.stem, path=path, **{g: data.get(g, {}) for g in GROUPS})
