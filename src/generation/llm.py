"""Ollama chat wrapper.

Owner: Sergio.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any


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


def chat(
    messages: list[dict[str, Any]],
    cfg,
    stream: bool = False,
    json: bool = False,
) -> str | Iterator[str]:
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
        return _message_content(response)

    def content_pieces() -> Iterator[str]:
        for chunk in response:
            piece = _message_content(chunk)
            if piece:
                yield piece

    return content_pieces()
