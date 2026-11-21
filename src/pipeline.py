# Run the build phase with one command, in the right order:
#   python -m src.pipeline build --config configs/final_system.yaml
#       read the PDFs -> slide tables -> course table -> chunks -> near-duplicates -> indexes
#   python -m src.pipeline all --config configs/final_system.yaml
#       the build, then the training of the router; it prints the command for the evaluation
# The image step (OCR and vision model) is run separately, once, because it takes hours:
#   python -m src.vision.run_vision   and then   python -m src.vision.describe
from __future__ import annotations

import argparse
import sys
import time

from src.config import load_config


# Run one step and print its result and how long it took. A step that is not written yet
# (NotImplementedError) is skipped with a message instead of stopping the whole build.
def _step(name: str, fn, *args) -> None:
    start = time.perf_counter()
    try:
        out = fn(*args)
    except NotImplementedError as exc:
        print(f"SKIP  {name}: {exc}")
        return
    print(f"DONE  {name}: {out}  ({time.perf_counter() - start:.1f}s)")


# Build the knowledge base of a configuration. Each step reads the files written by the step before it.
def build(cfg) -> None:
    from src.kb import build_indexes, chunking, course_table, dedup, parse_slides, parse_syllabi, slide_tables

    print(f"Building knowledge base for {cfg.name} (kb_sig={cfg.kb_sig})")
    _step("parse_syllabi", parse_syllabi.parse_syllabi, cfg)
    _step("parse_slides", parse_slides.parse_slides, cfg)
    # The slide tables only with kb.slide_tables (final_system.yaml).
    if cfg.kb.get("slide_tables", False):
        _step("slide_tables", slide_tables.build_slide_tables, cfg)
    # The course table is built before the chunks, because each course also gets a summary chunk made from it.
    _step("course_table", course_table.build_course_table, cfg)
    _step("chunking", chunking.build_chunks, cfg)
    # Near-duplicate removal, on by default (kb.dedup in base.yaml).
    if cfg.kb.get("dedup", True):
        _step("dedup", dedup.dedup, cfg)
    _step("build_indexes", build_indexes.build_indexes, cfg)


# The build plus the training of the router. The evaluation is run on its own afterwards, because it needs
# Ollama and takes much longer.
def run_all(cfg) -> None:
    from src.train import train_router

    build(cfg)
    _step("train_router", train_router.train, cfg)
    print("Then: python -m eval.run_eval --config", cfg.path.name, "--split dev")


# Read the command ("build" or "all") and the config, and run it.
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "all"])
    parser.add_argument("--config", default="configs/final_system.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    {"build": build, "all": run_all}[args.command](cfg)
    return 0


# Runs only when the file is executed directly.
if __name__ == "__main__":
    sys.exit(main())
