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
} # defining file paths


def chunks_path(kb_sig: str) -> Path: # builds path for saved chunk records belonging to KB version
    return PATHS["processed"] / f"chunks__{kb_sig}.jsonl"


def parents_path(kb_sig: str) -> Path: # builds path for saved chunk parent records
    return PATHS["processed"] / f"parents__{kb_sig}.jsonl"


def bm25_path(kb_sig: str) -> Path: # builds path for saved chunk BM25 keyword index
    return PATHS["processed"] / f"bm25__{kb_sig}.pkl"


GROUPS = ("kb", "retrieval", "routing", "generation", "eval") # list of YAML sections


@dataclass # automatically creates __init__ for fields below
class Config: # declaring the shape of a "Config" object (instances will hold configurations)
    name: str                                                   # required field (name will be text)
    path: Path                                                  # required field (holds the path to the config file)
    kb: dict[str, Any] = field(default_factory=dict)            # field for kb settings of YAML (creates a new dictionary if none provided)
    retrieval: dict[str, Any] = field(default_factory=dict)     # same as above but for retrieval  settings
    routing: dict[str, Any] = field(default_factory=dict)       # same as above but for routing    settings
    generation: dict[str, Any] = field(default_factory=dict)    # same as above but for generation settings
    eval: dict[str, Any] = field(default_factory=dict)          # same as above but for eval       settings

    @property # makes the method below have attribute behavior (ex: cfg.kb_sig , not cgf.kb_sig() )
    def kb_sig(self) -> str: # result of the config is intended as text
        blob = json.dumps(self.kb, sort_keys=True, default=str)         # takes kb dictionary of config and converts to a string in JSON format
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]       # turns the string into bytes

    @property # attribute behavior
    def collection_name(self) -> str:
        return f"kb_{self.kb_sig}"     # adds "kb_" to the string output of kb_sig (above) 


def _merge(base: dict, override: dict) -> dict: # combines inherited settings with overrides
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_raw(path: Path) -> dict: # reads a YAML file
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    if parent:
        data = _merge(_load_raw(path.parent / parent), data)
    return data


def load_config(path: str | Path = "configs/base.yaml") -> Config: # callable function for loading a Config object from a YAML file
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    data = _load_raw(path)
    unknown = set(data) - set(GROUPS)
    if unknown:
        raise ValueError(f"{path.name}: unknown config groups {sorted(unknown)}")
    return Config(name=path.stem, path=path, **{g: data.get(g, {}) for g in GROUPS})
