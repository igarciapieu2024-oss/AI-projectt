from types import SimpleNamespace

import pytest

import src.warmup as warmup


def test_warm_up_loads_enabled_retrieval_models(monkeypatch):
    calls = []
    monkeypatch.setattr(warmup.embed, "warm_up", lambda model: calls.append(("embed", model)))
    monkeypatch.setattr(warmup.rerank, "warm_up", lambda cfg: calls.append(("rerank", cfg)))
    monkeypatch.setattr(warmup.router, "warm_up", lambda cfg: calls.append(("router", cfg)))
    cfg = SimpleNamespace(
        kb={"embed_model": "local-embedder"},
        retrieval={"rerank": True},
        routing={"router": "classifier"},
    )

    warmup.warm_up(cfg)

    assert calls == [
        ("embed", "local-embedder"),
        ("rerank", cfg),
        ("router", cfg),
    ]


def test_warm_up_requires_configured_embedder(monkeypatch):
    cfg = SimpleNamespace(kb={}, retrieval={}, routing={})

    with pytest.raises(ValueError, match="kb.embed_model"):
        warmup.warm_up(cfg)
