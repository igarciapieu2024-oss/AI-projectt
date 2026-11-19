"""One command per job.

    python -m src.pipeline build --config configs/final.yaml   # parse -> chunk -> dedup -> indexes
    python -m src.pipeline all   --config configs/final.yaml   # build + course table + router + evaluation + figures

The image step runs separately, once:  python -m src.vision.run_vision
Owner: Francesca (skeleton); each step belongs to its module's owner.
"""
from __future__ import annotations

import argparse
import sys
import time

from src.config import load_config


def _step(name: str, fn, *args) -> None: # runs a task and prints how long it took ("SKIP" if error)
    start = time.perf_counter()
    try:
        out = fn(*args)
    except NotImplementedError as exc:
        print(f"SKIP  {name}: {exc}")
        return
    print(f"DONE  {name}: {out}  ({time.perf_counter() - start:.1f}s)")


def build(cfg) -> None: # prepares the searchable course knowledge base
    from src.kb import build_indexes, chunking, course_table, dedup, parse_slides, parse_syllabi, slide_tables

    print(f"Building knowledge base for {cfg.name} (kb_sig={cfg.kb_sig})")
    _step("parse_syllabi", parse_syllabi.parse_syllabi, cfg) # parses syllabi
    _step("parse_slides", parse_slides.parse_slides, cfg)
    if cfg.kb.get("slide_tables", False):                    # extract slide tables
        _step("slide_tables", slide_tables.build_slide_tables, cfg)
    # the course table is built before the chunks, because each course also gets a summary chunk made from it
    _step("course_table", course_table.build_course_table, cfg) # build course table
    _step("chunking", chunking.build_chunks, cfg)               # turning material into chunks
    if cfg.kb.get("dedup", True):                               # marks duplicates
        _step("dedup", dedup.dedup, cfg)
    _step("build_indexes", build_indexes.build_indexes, cfg)    # builds search indices


def run_all(cfg) -> None: # runs build(cfg) then trains the question router
    from src.train import train_router

    build(cfg)  # includes the course table
    _step("train_router", train_router.train, cfg)
    print("Then: python -m eval.run_eval --config", cfg.path.name, "--split dev")


def main(argv: list[str] | None = None) -> int: # handles command line use
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "all"])
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    {"build": build, "all": run_all}[args.command](cfg)
    return 0


if __name__ == "__main__": # runs main when the file is launched as a command
    sys.exit(main())
