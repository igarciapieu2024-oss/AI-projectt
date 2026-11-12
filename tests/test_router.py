from types import ModuleType, SimpleNamespace

import src.retrieval.router as router_module
from src.schemas import RouteDecision


def _cfg(**routing):
    return SimpleNamespace(routing=routing, generation={"llm_model": "test"})


def test_none_mode_returns_none_decision():
    decision = router_module.route("Question", _cfg(router="none"))

    assert decision == RouteDecision(label="none", prob=0.0, source="none")


def test_classifier_mode_uses_cached_model_prediction(monkeypatch, tmp_path):
    model_path = tmp_path / "router.joblib"
    model_path.touch()
    monkeypatch.setitem(router_module.PATHS, "router_model", model_path)

    class FakeRouter:
        def route(self, question):
            assert question == "Question"
            return "course_admin", 0.9

    monkeypatch.setattr(router_module, "_load_router", lambda path: FakeRouter())

    decision = router_module.route("Question", _cfg(router="classifier", min_prob=0.6))

    assert decision == RouteDecision(label="course_admin", prob=0.9, source="classifier")


def test_classifier_low_probability_falls_back_to_llm(monkeypatch, tmp_path):
    model_path = tmp_path / "router.joblib"
    model_path.touch()
    monkeypatch.setitem(router_module.PATHS, "router_model", model_path)
    monkeypatch.setattr(
        router_module,
        "_load_router",
        lambda path: SimpleNamespace(route=lambda question: ("course_content", 0.2)),
    )
    train_router = ModuleType("src.train.train_router")
    train_router.LLM_PROMPT = "Classify"
    train_router.parse_llm_label = lambda response: response.strip()
    monkeypatch.setitem(__import__("sys").modules, "src.train.train_router", train_router)
    monkeypatch.setattr(router_module, "chat", lambda messages, cfg: "cross_course")

    decision = router_module.route("Compare courses", _cfg(router="classifier", min_prob=0.6))

    assert decision == RouteDecision(label="cross_course", prob=1.0, source="llm")


def test_missing_classifier_model_returns_none(monkeypatch, tmp_path):
    monkeypatch.setitem(router_module.PATHS, "router_model", tmp_path / "missing.joblib")

    decision = router_module.route("Question", _cfg(router="classifier"))

    assert decision.label == "none"
    assert decision.source == "none"
