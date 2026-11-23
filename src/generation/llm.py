# Send messages to the local language model through Ollama and return its reply. chat() is the only
# place where the project talks to the LLM: answer.py uses it to write the answers, rewrite.py to rewrite
# follow-up questions, router.py when the classifier is unsure, and eval/judge.py for the judge model.
# Ollama runs on the laptop (http://127.0.0.1:11434), so no question or source leaves the computer.
from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

# The reply text, which behaves like a normal string but also keeps Ollama's done_reason ("stop", or
# "length" when the answer was cut by num_predict). answer.py uses it to trim a cut answer to its last
# complete sentence.
class ChatText(str):

    def __new__(cls, content: str, done_reason: str | None):
        response = super().__new__(cls, content)
        response.done_reason = done_reason
        return response


# The reply as a stream of text pieces, for the app, which shows the answer word by word. It also keeps
# the done_reason of the last piece.
class ChatStream(Iterator[str]):

    def __init__(self, response):
        self._response = iter(response)
        self.done_reason: str | None = None

    def __iter__(self) -> ChatStream:
        return self

    # Return the next piece of text; pieces without text (such as the final status message) are skipped.
    def __next__(self) -> str:
        while True:
            chunk = next(self._response)
            done_reason = _done_reason(chunk)
            if done_reason is not None:
                self.done_reason = done_reason
            piece = _message_content(chunk)
            if piece:
                return piece


# Get the text of a reply. Newer versions of the ollama library return an object and older ones a
# dictionary, so both are read.
def _message_content(response: Any) -> str:
    if isinstance(response, Mapping):
        message = response.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
    else:
        message = getattr(response, "message", None)
        content = getattr(message, "content", None)

    if not isinstance(content, str):
        raise ValueError("Ollama returned a response without message content")
    return content


# Get why the model stopped ("stop" or "length"), from an object or a dictionary; None if not given.
def _done_reason(response: Any) -> str | None:
    if isinstance(response, Mapping):
        value = response.get("done_reason")
    else:
        value = getattr(response, "done_reason", None)
    return value if isinstance(value, str) else None


# Send the messages to the model of the config (generation.llm_model) and return the reply text, or a
# stream of pieces when stream=True. json=True forces a JSON reply (used by the judge). num_predict
# replaces the token limit of the config for one call (rewrite.py uses a shorter one for questions).
def chat(
    messages: list[dict[str, Any]],
    cfg,
    stream: bool = False,
    json: bool = False,
    num_predict: int | None = None,
) -> ChatText | Iterator[str]:
    if not messages:
        raise ValueError("messages must contain at least one chat message")

    try:
        model = cfg.generation["llm_model"]
    except (AttributeError, KeyError) as error:
        raise ValueError("configuration must define generation.llm_model") from error

    # The generation settings sent to Ollama, only those the config defines.
    options = {}
    # temperature: how much randomness the model uses when it picks each word (0 in final_system.yaml:
    # always the most likely word).
    temperature = cfg.generation.get("temperature")
    if temperature is not None:
        options["temperature"] = temperature
    # seed: makes two runs of the same configuration give the same answers. With temperature 0.1 and no
    # seed, some answers changed between two runs of an unchanged system, which blurred the comparisons.
    seed = cfg.generation.get("seed")
    if seed is not None:
        options["seed"] = seed
    # The maximum number of tokens of the reply: the one given to this call, or generation.num_predict.
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

    # Imported here so the rest of the project works without the ollama library installed.
    import ollama

    # The local Ollama server; follow_redirects=False so the request can never be sent anywhere else.
    client = ollama.Client(host="http://127.0.0.1:11434", follow_redirects=False)
    response = client.chat(**request)
    if not stream:
        return ChatText(_message_content(response), _done_reason(response))
    return ChatStream(response)
