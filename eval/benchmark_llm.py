"""Compare local Ollama models on a fixed, cited-answer task.

Run from the repository root with:
    python -m eval.benchmark_llm
    python -m eval.benchmark_llm --models llama3.2:3b qwen2.5:7b --runs 3
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from src.config import load_config
from src.generation.llm import chat

DEFAULT_MODELS = ("llama3.1:8b", "qwen2.5:7b", "llama3.2:3b")
MAX_SECONDS = 10.0
MESSAGES = [
    {
        "role": "user",
        "content": (
            "Answer the question using only this course excerpt. Keep the answer to "
            "one sentence, include a citation to the excerpt, and do not add other "
            "facts.\n\n"
            "Question: What percentage of the grade is the final exam worth?\n"
            "Excerpt [S1]: The final exam is worth 40% of the final grade."
        ),
    }
]


@dataclass
class BenchmarkRun:
    model: str
    run_number: int
    elapsed_seconds: float
    response: str


def benchmark_model(model: str, runs: int) -> list[BenchmarkRun]:
    """Run the same prompt repeatedly and return elapsed times and responses."""
    config = load_config("configs/final.yaml")
    config.generation["llm_model"] = model
    results = []

    for run_number in range(1, runs + 1):
        started = time.perf_counter()
        response = chat(MESSAGES, config)
        elapsed = time.perf_counter() - started
        if not isinstance(response, str):
            raise TypeError("non-streaming chat() must return a string")
        results.append(BenchmarkRun(model, run_number, elapsed, response))

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Ollama model tags to test")
    parser.add_argument("--runs", type=int, default=1, help="Runs per model (default: 1)")
    parser.add_argument(
        "--limit-seconds",
        type=float,
        default=MAX_SECONDS,
        help="Target maximum time per answer (default: 10 seconds)",
    )
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.limit_seconds <= 0:
        parser.error("--limit-seconds must be greater than 0")

    print(f"Target: {args.limit_seconds:.1f} seconds per answer")
    print("Ollama must be running, and each requested model must already be downloaded.")
    passed = []
    for model in args.models:
        print(f"\n=== {model} ===")
        runs = benchmark_model(model, args.runs)
        model_passed = all(run.elapsed_seconds <= args.limit_seconds for run in runs)
        if model_passed:
            passed.append(model)
        for run in runs:
            status = "within target" if run.elapsed_seconds <= args.limit_seconds else "over target"
            print(f"Run {run.run_number}: {run.elapsed_seconds:.2f}s ({status})")
            print(f"Response: {run.response.strip()}")
        print(f"Timing result: {'PASS' if model_passed else 'FAIL'}")

    print("\nCompare both timing and whether the answer is accurate and cites [S1].")
    print(f"Models within the time target: {', '.join(passed) if passed else 'none'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
