"""Tests of the Streamlit app with the fake engine (owner: Irene).

Each test opens one page in a simulated browser (Streamlit's AppTest), clicks like a user would
and checks that the page shows what it should, without errors. Run with: pytest tests/test_app.py
They are skipped where Streamlit isn't installed (the GitHub check installs only the light libraries).
"""
import os
from pathlib import Path

import pytest

#1 setup: the fake engine, and skip everything if Streamlit isn't installed
os.environ["USE_FAKE"] = "1"  # set before the app is imported, so the pages use the fake engine
pytest.importorskip("streamlit")  # no Streamlit -> every test in this file is skipped, not failed
from streamlit.testing.v1 import AppTest  # noqa: E402  (after the skip, on purpose)

import app_content as content  # noqa: E402  (the example questions)

ROOT = Path(__file__).resolve().parent.parent  # the repo folder
TIMEOUT = 60  # seconds a page may take (the fake engine waits a little between words)


def open_page(name: str) -> AppTest:
    """Opens one page (e.g. "assistant") and runs it once, like opening it in the browser."""
    return AppTest.from_file(str(ROOT / "app_pages" / f"{name}.py"), default_timeout=TIMEOUT).run()


def no_errors(at: AppTest) -> None:
    """Fails the test if the page showed a Python error."""
    assert not at.exception, [e.value for e in at.exception]


#2 every page opens without errors

def test_whole_app_starts_on_home():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT).run()  # the entry point with the menu
    no_errors(at)


@pytest.mark.parametrize("page", ["home", "assistant", "compare", "how_it_works"])
def test_page_opens(page):
    no_errors(open_page(page))


#3 the chat: a question gets an answer with sources, and an out-of-scope one is refused

def test_example_question_gets_cited_answer():
    at = open_page("assistant")
    at.button(key="start_0").click().run()  # clicks the first example question card
    no_errors(at)
    messages = at.session_state.messages
    assert [m["role"] for m in messages] == ["user", "assistant"]  # the question and its answer
    result = messages[-1]["result"]
    assert result.context_hits and "[S" in result.answer_text  # sources, and the answer cites them


def test_out_of_scope_question_is_refused():
    at = open_page("assistant")
    at.chat_input[0].set_value(content.EXAMPLE_QUESTIONS[-1]).run()  # "When is the end-of-year party?"
    no_errors(at)
    assert at.session_state.messages[-1]["result"].refused  # the system says it can't answer instead of guessing


def test_follow_up_sends_history():
    at = open_page("assistant")
    at.chat_input[0].set_value(content.EXAMPLE_QUESTIONS[0]).run()
    at.chat_input[0].set_value("And is there a minimum grade?").run()  # a follow-up
    no_errors(at)
    assert len(at.session_state.messages) == 4  # 2 questions, 2 answers


#4 when the engine fails, the page explains it instead of crashing

def test_engine_failure_shows_a_readable_message(monkeypatch):
    import app_components.answer_view as view

    def broken_engine(*args, **kwargs):  # behaves like the real engine before it's written
        raise NotImplementedError("answer_stream: not built yet")
        yield  # never reached; makes this a generator, like the real one

    monkeypatch.setattr(view, "answer_stream", broken_engine)
    at = open_page("assistant")
    at.chat_input[0].set_value("What is the final exam worth?").run()
    no_errors(at)  # no red traceback on the page
    assert at.session_state.messages[-1].get("error")  # saved as a failed turn, so it doesn't retry on every click


def test_error_messages_say_how_to_fix():
    from app_components.answer_view import explain_error

    assert "Ollama" in explain_error(ConnectionError("connection refused"))[1]  # Ollama closed -> open the app
    assert "pipeline build" in explain_error(FileNotFoundError("index"))[1]  # no index -> build it
    assert "USE_FAKE" in explain_error(NotImplementedError())[1]  # engine not written -> use the fake one


#5 compare: both answers appear, and only ours has sources

def test_compare_shows_both_answers():
    at = open_page("compare")
    at.button[0].click().run()  # the "Compare" button
    no_errors(at)
    assert len(at.columns) >= 2  # the two panels
