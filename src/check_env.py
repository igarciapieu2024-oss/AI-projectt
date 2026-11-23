# Check that this laptop is ready to run the project: the Python version, the libraries, the Ollama models
# of the final system and the folders the project writes to. Every check prints OK or FAIL with how to fix it.
# Run from the repo folder:  python -m src.check_env
from __future__ import annotations

import importlib
import sys

from src.config import PATHS, load_config

# The libraries the project imports (some, such as open_clip and imagehash, are only needed by earlier
# experiments; they are kept in the list because they are in requirements.txt).
LIBRARIES = [
    "yaml", "numpy", "pandas", "pymupdf", "chromadb", "sentence_transformers", "rank_bm25",
    "rapidfuzz", "ollama", "streamlit", "sklearn", "torch", "torchvision", "open_clip",
    "easyocr", "imagehash", "PIL", "pytest",
]


# Run all the checks; returns 0 if everything is ready and 1 otherwise.
def main() -> int:
    ok = True

    # Print one check and remember if any failed.
    def report(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"{'OK  ' if passed else 'FAIL'}  {name}{('  -  ' + detail) if detail else ''}")

    report("Python 3.11+", sys.version_info >= (3, 11), sys.version.split()[0])

    for lib in LIBRARIES:
        try:
            importlib.import_module(lib)
            report(f"import {lib}", True)
        except Exception as exc:  # noqa: BLE001
            report(f"import {lib}", False, f"{type(exc).__name__}: run pip install -r requirements.txt")

    # The Ollama models of the final system: the answer model, the vision model and the judge model.
    cfg = load_config("configs/final_system.yaml")
    wanted = {cfg.generation.get("llm_model"), cfg.generation.get("vision_model"), cfg.eval.get("judge_model")} - {None}
    try:
        import ollama

        # The models installed in Ollama. Different versions of the library return a dictionary or an object,
        # and a model counts as installed with or without its size tag (":8b").
        listed = ollama.list()
        models = listed.get("models", []) if isinstance(listed, dict) else getattr(listed, "models", [])
        names = set()
        for m in models:
            name = m.get("model") or m.get("name") if isinstance(m, dict) else getattr(m, "model", "")
            names.add(name)
            names.add(name.split(":")[0])
        for model in sorted(wanted):
            report(f"ollama model {model}", model in names or model.split(":")[0] in names, f"ollama pull {model}")
    except Exception as exc:  # noqa: BLE001
        report("Ollama running", False, f"{type(exc).__name__}: open the Ollama app")

    # The folders the build and the evaluation write to: write and delete a small test file in each.
    for key in ("chroma", "interim", "processed", "reports"):
        path = PATHS[key]
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_test"
            probe.write_text("ok")
            probe.unlink()
            report(f"writable {path.relative_to(PATHS['root'])}/", True)
        except OSError as exc:
            report(f"writable {path}", False, str(exc))

    print("\nAll OK" if ok else "\nFix the FAIL lines above, then run this again.")
    return 0 if ok else 1


# Runs only when the file is executed directly.
if __name__ == "__main__":
    sys.exit(main())
