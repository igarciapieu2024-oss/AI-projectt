# Compare the speed of local Ollama models on one fixed task: answer a short question from one excerpt and cite it. We
# used it to choose the answer model: a model has to answer within the time target to be usable live. Speed alone says
# nothing about accuracy, so each answer is also printed, to check it by eye.
#   python -m eval.benchmark_llm
#   python -m eval.benchmark_llm --models llama3.2:3b qwen2.5:7b --runs 3
from __future__ import annotations

# argparse reads the options of the command, time measures each answer and dataclass defines the result of one run.
import argparse
import time
from dataclasses import dataclass

# The config reader, and chat(), which sends the prompt to the local LLM.
from src.config import load_config
from src.generation.llm import chat

# The models compared by default and the time target per answer (in seconds).
DEFAULT_MODELS = ("llama3.1:8b", "qwen2.5:7b", "llama3.2:3b")
MAX_SECONDS = 10.0
# The same prompt for every model and every run, so the times can be compared.
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


# The result of one run: the model, the run number, the time it took and its answer.
@dataclass
class BenchmarkRun:
    model: str
    run_number: int
    elapsed_seconds: float
    response: str


# Run the fixed prompt several times with one model and time each answer. We use the settings of the final system,
# with only the model replaced.
def benchmark_model(model: str, runs: int) -> list[BenchmarkRun]:
    # The settings of the final system, with only the model changed.
    config = load_config("configs/final_system.yaml")
    config.generation["llm_model"] = model
    # The results of every run of this model.
    results = []

    for run_number in range(1, runs + 1):
        # Time one answer.
        started = time.perf_counter()
        response = chat(MESSAGES, config)
        elapsed = time.perf_counter() - started
        # Without streaming, chat() has to give back text.
        if not isinstance(response, str):
            raise TypeError("non-streaming chat() must return a string")
        results.append(BenchmarkRun(model, run_number, elapsed, response))

    return results


# Run every model, print each answer with its time, and list the models that stayed within the target. It returns 0 if
# at least one model passed.
def main(argv: list[str] | None = None) -> int:
    # The options: which models, how many runs of each and the time target.
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
    # Both numbers have to be positive.
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.limit_seconds <= 0:
        parser.error("--limit-seconds must be greater than 0")

    print(f"Target: {args.limit_seconds:.1f} seconds per answer")
    print("Ollama must be running, and each requested model must already be downloaded.")
    # The models that stayed within the target.
    passed = []
    for model in args.models:
        print(f"\n=== {model} ===")
        runs = benchmark_model(model, args.runs)
        # A model passes only if every one of its runs stayed within the target.
        model_passed = all(run.elapsed_seconds <= args.limit_seconds for run in runs)
        if model_passed:
            passed.append(model)
        # Print each run with its time and its answer.
        for run in runs:
            status = "within target" if run.elapsed_seconds <= args.limit_seconds else "over target"
            print(f"Run {run.run_number}: {run.elapsed_seconds:.2f}s ({status})")
            print(f"Response: {run.response.strip()}")
        print(f"Timing result: {'PASS' if model_passed else 'FAIL'}")

    # The exit code is 0 when at least one model passed.
    print("\nCompare both timing and whether the answer is accurate and cites [S1].")
    print(f"Models within the time target: {', '.join(passed) if passed else 'none'}")
    return 0 if passed else 1


# Runs only when the file is executed directly.
if __name__ == "__main__":
    raise SystemExit(main())
