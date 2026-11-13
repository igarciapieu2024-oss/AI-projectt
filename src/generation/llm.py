"""Ollama chat wrapper.

Owner: Sergio.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any


class ChatText(str):
    """Text response that retains Ollama's completion reason."""

    def __new__(cls, content: str, done_reason: str | None):
        response = super().__new__(cls, content)
        response.done_reason = done_reason
        return response


class ChatStream(Iterator[str]):
    """String iterator that exposes the last Ollama completion reason."""

    def __init__(self, response):
        self._response = iter(response)
        self.done_reason: str | None = None

    def __iter__(self) -> ChatStream:
        return self

    def __next__(self) -> str:
        while True:
            chunk = next(self._response)
            done_reason = _done_reason(chunk)
            if done_reason is not None:
                self.done_reason = done_reason
            piece = _message_content(chunk)
            if piece:
                return piece


def _message_content(response: Any) -> str:
    """Get message content from Ollama's object or dictionary response."""
    if isinstance(response, Mapping):
        message = response.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
    else:
        message = getattr(response, "message", None)
        content = getattr(message, "content", None)

    if not isinstance(content, str):
        raise ValueError("Ollama returned a response without message content")
    return content


def _done_reason(response: Any) -> str | None:
    if isinstance(response, Mapping):
        value = response.get("done_reason")
    else:
        value = getattr(response, "done_reason", None)
    return value if isinstance(value, str) else None


def chat(
    messages: list[dict[str, Any]],
    cfg,
    stream: bool = False,
    json: bool = False,
    num_predict: int | None = None,
) -> ChatText | Iterator[str]:
    """Return the reply text, or an iterator of text pieces when stream=True."""
    if not messages:
        raise ValueError("messages must contain at least one chat message")

    try:
        model = cfg.generation["llm_model"]
    except (AttributeError, KeyError) as error:
        raise ValueError("configuration must define generation.llm_model") from error

    options = {}
    temperature = cfg.generation.get("temperature")
    if temperature is not None:
        options["temperature"] = temperature
    configured_limit = cfg.generation.get("num_predict")
    token_limit = num_predict if num_predict is not None else configured_limit
    if token_limit is not None:
        if isinstance(token_limit, bool) or not isinstance(token_limit, int) or token_limit < 1:
            setting = "num_predict" if num_predict is not None else "generation.num_predict"
            raise ValueError(
                f"{setting} must be a positive integer; got {token_limit!r}"
            )
        options["num_predict"] = token_limit

    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if options:
        request["options"] = options
    if json:
        request["format"] = "json"

    import ollama

    client = ollama.Client(host="http://127.0.0.1:11434", follow_redirects=False)
    response = client.chat(**request)
    if not stream:
        return ChatText(_message_content(response), _done_reason(response))
    return ChatStream(response)
