import sys
from types import SimpleNamespace

import pytest

from src.generation.llm import chat


def _config():
    return SimpleNamespace(
        generation={"llm_model": "local-model", "temperature": 0.25}
    )


def test_chat_uses_loopback_even_when_ollama_host_is_remote(monkeypatch):
    calls = []
    monkeypatch.setenv("OLLAMA_HOST", "https://example.com")

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(("client", kwargs))

        def chat(self, **kwargs):
            calls.append(("chat", kwargs))
            return {"message": {"content": "Hello"}}

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=FakeClient))

    assert chat([{"role": "user", "content": "Hi"}], _config()) == "Hello"
    assert calls == [
        ("client", {"host": "http://127.0.0.1:11434", "follow_redirects": False}),
        ("chat", {
            "model": "local-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "options": {"temperature": 0.25},
        }),
    ]


def test_chat_streams_text_pieces(monkeypatch):
    chunks = iter([
        {"message": {"content": "Hello"}},
        {"message": {"content": " world"}},
        {"message": {"content": ""}},
    ])

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def chat(self, **kwargs):
            return chunks

    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(Client=FakeClient),
    )

    result = chat([{"role": "user", "content": "Hi"}], _config(), stream=True)

    assert iter(result) is result
    assert list(result) == ["Hello", " world"]


def test_chat_json_mode_requests_json_format(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def chat(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))

    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(Client=FakeClient),
    )

    assert chat([{"role": "user", "content": "Return JSON"}], _config(), json=True) == '{"ok": true}'
    assert calls[0]["format"] == "json"


def test_chat_rejects_empty_messages_and_missing_model(monkeypatch):
    with pytest.raises(ValueError, match="at least one"):
        chat([], _config())
    with pytest.raises(ValueError, match="llm_model"):
        chat([{"role": "user", "content": "Hi"}], SimpleNamespace(generation={}))


def test_chat_rejects_malformed_ollama_response(monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def chat(self, **kwargs):
            return {"message": {}}

    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(Client=FakeClient),
    )

    with pytest.raises(ValueError, match="without message content"):
        chat([{"role": "user", "content": "Hi"}], _config())
