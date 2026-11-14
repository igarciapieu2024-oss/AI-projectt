from types import SimpleNamespace

import pytest

import src.warmup as warmup


def test_warm_up_loads_enabled_retrieval_models(monkeypatch):
    from src.config import PATHS, bm25_path, chunks_path

    calls = []
    monkeypatch.setattr(
        warmup.index_cache,
        "get_chroma_collection",
        lambda path, name: calls.append(("chroma", path, name)),
    )
    monkeypatch.setattr(
        warmup.index_cache,
        "load_bm25",
        lambda path: calls.append(("bm25", path)),
    )
    monkeypatch.setattr(
        warmup.index_cache,
        "load_chunks",
        lambda path: calls.append(("chunks", path)),
    )
    monkeypatch.setattr(warmup.embed, "warm_up", lambda model: calls.append(("embed", model)))
    monkeypatch.setattr(warmup.rerank, "warm_up", lambda cfg: calls.append(("rerank", cfg)))
    monkeypatch.setattr(warmup.router, "warm_up", lambda cfg: calls.append(("router", cfg)))
    monkeypatch.setattr(
        warmup,
        "chat",
        lambda messages, cfg, **kwargs: calls.append(("llm", messages, cfg, kwargs)),
    )
    cfg = SimpleNamespace(
        kb={"embed_model": "local-embedder"},
        retrieval={"rerank": True},
        routing={"router": "classifier"},
        kb_sig="testkb00",
        collection_name="kb_testkb00",
    )

    warmup.warm_up(cfg)

    assert calls == [
        ("chroma", PATHS["chroma"], "kb_testkb00"),
        ("bm25", bm25_path("testkb00")),
        ("chunks", chunks_path("testkb00")),
        ("embed", "local-embedder"),
        ("rerank", cfg),
        ("router", cfg),
        ("llm", [{"role": "user", "content": "Hi"}], cfg, {"num_predict": 1}),
    ]


def test_warm_up_requires_configured_embedder(monkeypatch):
    cfg = SimpleNamespace(kb={}, retrieval={}, routing={})

    with pytest.raises(ValueError, match="kb.embed_model"):
        warmup.warm_up(cfg)
