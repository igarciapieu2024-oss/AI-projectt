
"""Tests for the evaluation harness using fake answers.

Owner: Elena.
"""
import csv
import json

import pytest

from eval.run_eval import evaluate


@pytest.fixture
def sample_questions(tmp_path):
    """Create a temporary CSV without modifying the real dataset."""
    path = tmp_path / "questions.csv"

    rows = [
        {
            "question_id": "EL01",
            "split": "dev",
            "type": "admin",
            "turn1": "",
            "question": "What is the final exam worth?",
            "course_code": "c01",
            "gold_ids": "c01_syl_sec04",
            "reference_answer": "40%",
            "key_facts": "final exam 40%",
            "answerable": "1",
            "author": "Elena",
        },
        {
            "question_id": "EL02",
            "split": "dev",
            "type": "unanswerable",
            "turn1": "",
            "question": "What is the weather tomorrow?",
            "course_code": "c01",
            "gold_ids": "",
            "reference_answer": "",
            "key_facts": "",
            "answerable": "0",
            "author": "Elena",
        },
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return path


def test_evaluate_fake_answers(sample_questions, tmp_path, monkeypatch):
    """The evaluator processes questions and saves correct results."""
    monkeypatch.setitem(
        __import__("eval.run_eval", fromlist=["PATHS"]).PATHS,
        "root",
        tmp_path,
    )

    output = evaluate(
        config_path="configs/b1_v1.yaml",
        split="dev",
        use_fake=True,
        questions_path=sample_questions,
    )

    assert output.exists()

    results = json.loads(output.read_text(encoding="utf-8"))

    assert len(results) == 2

    normal = results[0]
    assert normal["question_id"] == "EL01"
    assert normal["retrieval_hit"] is True
    assert normal["refused"] is False
    assert normal["refusal_correct"] is True
    assert len(normal["citations"]) == 5

    unanswerable = results[1]
    assert unanswerable["question_id"] == "EL02"
    assert unanswerable["refused"] is True
    assert unanswerable["refusal_correct"] is True


def test_test_split_is_locked(sample_questions):
    """Prevent accidental evaluation of the locked test set."""
    with pytest.raises(ValueError, match="locked"):
        evaluate(
            config_path="configs/b1_v1.yaml",
            split="test",
            use_fake=True,
            questions_path=sample_questions,
        )

def test_evaluation_exports(sample_questions, tmp_path, monkeypatch):
    """Evaluation generates individual, summary, and CSV reports."""
    from eval import run_eval

    monkeypatch.setitem(run_eval.PATHS, "root", tmp_path)

    output = run_eval.evaluate(
        config_path="configs/b1_v1.yaml",
        split="dev",
        use_fake=True,
        questions_path=sample_questions,
    )

    # Individual results
    assert output.exists()

    results = json.loads(output.read_text(encoding="utf-8"))
    assert len(results) == 2
    assert results[0]["recall_at_5"] == 1.0
    assert results[1]["recall_at_5"] is None

    # Aggregate results
    summary_path = output.parent / "summary.json"
    assert summary_path.exists()

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    assert summary["n_questions"] == 2
    assert summary["n_retrieval_scored"] == 1
    assert summary["fake_mode"] is True
    assert summary["recall_at_5"] == 1.0
    assert summary["refusal_accuracy"] == 1.0

    # Experiment comparison report
    report_path = tmp_path / "reports" / "ablation_dev.csv"
    assert report_path.exists()

    with report_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["config"] == "b1_v1"
    assert rows[0]["fake_mode"] == "True"
    assert float(rows[0]["recall_at_5"]) == 1.0

def test_judge_integration(sample_questions, tmp_path, monkeypatch):
    """Verify that judge results are included in evaluation reports."""
    from eval import run_eval

    monkeypatch.setitem(run_eval.PATHS, "root", tmp_path)

    # Simulate the real answer engine using the existing fake.
    from src.generation import fake

    import sys
    import types

    fake_answer_module = types.ModuleType("src.generation.answer")
    fake_answer_module.answer = fake.answer

    monkeypatch.setitem(
        sys.modules,
        "src.generation.answer",
        fake_answer_module,
    )

    def simulated_judge(
        answer_text, reference_answer, sources, cfg
    ):
        return {
            "correctness": 4,
            "claims": [
                {"text": "Example claim", "supported": True},
                {"text": "Unsupported claim", "supported": False},
            ],
            "n_claims": 2,
            "unsupported_claims": ["Unsupported claim"],
            "n_unsupported_claims": 1,
            "hallucination_rate": 0.5,
        }

    output = run_eval.evaluate(
        config_path="configs/b1_v1.yaml",
        split="dev",
        use_fake=False,
        questions_path=sample_questions,
        use_judge=True,
        judge_fn=simulated_judge,
    )

    results = json.loads(output.read_text(encoding="utf-8"))

    assert len(results) == 2
    assert results[0]["judge"]["correctness"] == 4

    summary = json.loads(
        (output.parent / "summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["n_judged"] == 2
    assert summary["mean_correctness"] == 4
    assert summary["hallucination_rate"] == 0.5

    report_path = tmp_path / "reports" / "ablation_dev.csv"

    with report_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert float(rows[0]["mean_correctness"]) == 4.0
    assert float(rows[0]["hallucination_rate"]) == 0.5
