# Tests of the speed benchmark of the local models (eval/benchmark_llm.py), with a fake chat function.
from eval import benchmark_llm


# Each run keeps the answer and the time it took.
def test_benchmark_model_records_response_and_elapsed_time(monkeypatch):
    class Config:
        generation = {"llm_model": "initial-model"}

    monkeypatch.setattr(benchmark_llm, "load_config", lambda _: Config())
    monkeypatch.setattr(benchmark_llm, "chat", lambda messages, cfg: f"answer from {cfg.generation['llm_model']}")

    results = benchmark_llm.benchmark_model("test-model", runs=2)

    assert [result.model for result in results] == ["test-model", "test-model"]
    assert [result.run_number for result in results] == [1, 2]
    assert all(result.elapsed_seconds >= 0 for result in results)
    assert all(result.response == "answer from test-model" for result in results)


# The models within the time target are reported, and the exit code is 0.
def test_main_reports_models_that_meet_the_time_target(monkeypatch, capsys):
    monkeypatch.setattr(
        benchmark_llm,
        "benchmark_model",
        lambda model, runs: [
            benchmark_llm.BenchmarkRun(model, 1, 2.0, "The final exam is 40% [S1].")
        ],
    )

    assert benchmark_llm.main(["--models", "test-model"]) == 0
    output = capsys.readouterr().out
    assert "Timing result: PASS" in output
    assert "test-model" in output


# When every model is over the target, the exit code is 1.
def test_main_returns_failure_when_all_models_exceed_target(monkeypatch, capsys):
    monkeypatch.setattr(
        benchmark_llm,
        "benchmark_model",
        lambda model, runs: [
            benchmark_llm.BenchmarkRun(model, 1, 11.0, "late response")
        ],
    )

    assert benchmark_llm.main(["--models", "test-model"]) == 1
    assert "Timing result: FAIL" in capsys.readouterr().out
