# Tests of the only function that talks to Ollama (src/generation/llm.py), with a fake Ollama client.
import sys
from types import SimpleNamespace

import pytest

from src.generation.llm import chat


# A test configuration with a model name.
def _config():
    return SimpleNamespace(
        generation={"llm_model": "local-model", "temperature": 0.25}
    )


# The request always goes to the local Ollama, even if another host is set on the computer.
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


# With streaming, the answer arrives in pieces.
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


# json=True asks Ollama for a JSON reply.
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


# The token limit of the config is sent to Ollama.
def test_chat_uses_configured_answer_token_limit(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def chat(self, **kwargs):
            calls.append(kwargs)
            return {"message": {"content": "Short answer"}}

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=FakeClient))
    cfg = _config()
    cfg.generation["num_predict"] = 400

    assert chat([{"role": "user", "content": "Question"}], cfg) == "Short answer"
    assert calls[0]["options"] == {"temperature": 0.25, "num_predict": 400}


# A call can use its own, smaller token limit.
def test_chat_allows_small_query_token_limit_override(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def chat(self, **kwargs):
            calls.append(kwargs)
            return {"message": {"content": "Short query"}}

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=FakeClient))
    cfg = _config()
    cfg.generation["num_predict"] = 400

    assert chat([{"role": "user", "content": "Translate"}], cfg, num_predict=96) == "Short query"
    assert calls[0]["options"]["num_predict"] == 96


# The reply keeps why the model stopped, for cut answers.
def test_chat_retains_completion_reason_for_truncated_responses(monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def chat(self, **kwargs):
            return {
                "message": {"content": "Complete. [S1] Cut off"},
                "done_reason": "length",
            }

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=FakeClient))

    response = chat([{"role": "user", "content": "Question"}], _config())

    assert response == "Complete. [S1] Cut off"
    assert response.done_reason == "length"


# The streaming reply keeps it too.
def test_chat_stream_retains_final_completion_reason(monkeypatch):
    chunks = iter([
        {"message": {"content": "Complete. [S1] "}},
        {"message": {"content": "Cut off"}},
        {"message": {"content": ""}, "done_reason": "length"},
    ])

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def chat(self, **kwargs):
            return chunks

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=FakeClient))
    response = chat([{"role": "user", "content": "Question"}], _config(), stream=True)

    assert list(response) == ["Complete. [S1] ", "Cut off"]
    assert response.done_reason == "length"


# A wrong token limit raises an error.
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "256"])
def test_chat_rejects_invalid_answer_token_limit(monkeypatch, value):
    cfg = _config()
    cfg.generation["num_predict"] = value
    with pytest.raises(ValueError, match="generation.num_predict"):
        chat([{"role": "user", "content": "Question"}], cfg)


# No messages, or no model in the config, raises an error.
def test_chat_rejects_empty_messages_and_missing_model(monkeypatch):
    with pytest.raises(ValueError, match="at least one"):
        chat([], _config())
    with pytest.raises(ValueError, match="llm_model"):
        chat([{"role": "user", "content": "Hi"}], SimpleNamespace(generation={}))


# A reply without text raises an error.
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


# The seed of the config is sent next to the temperature.
def test_chat_passes_the_configured_seed(monkeypatch):
    # 1. a fixed seed reaches Ollama next to the temperature, so reruns give the same answers
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def chat(self, **kwargs):
            calls.append(kwargs)
            return {"message": {"content": "Same answer"}}

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=FakeClient))
    cfg = _config()
    cfg.generation["seed"] = 42
    chat([{"role": "user", "content": "Question"}], cfg)
    assert calls[0]["options"] == {"temperature": 0.25, "seed": 42}
