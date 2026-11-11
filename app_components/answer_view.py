"""How an answer is written on screen: word by word, with the [S1] citations as yellow chips (owner: Irene)."""
from __future__ import annotations  # lets us write types like "list[dict] | None" in any Python version

import html as html_lib  # escapes < and > in the answer, so the LLM's text can't break our HTML
import re  # finds the [S1], [S2]... citations

import streamlit as st  # the library that builds the page

import app_content as content  # the product name
from app_components.engine import answer_stream  # the fake or the real engine
from app_components.theme import html, icon  # our HTML helper and the icons
from src.schemas import AnswerResult, Filters  # the agreed formats of an answer and of the filters

_CITATION = re.compile(r"\[(S\d+)\]")  # a citation in the answer text, e.g. [S3]


#1 the answer text with citation chips

def format_answer(text: str) -> str:
    """The answer as HTML: the text escaped, and every [S1] turned into a yellow chip."""
    safe = html_lib.escape(text)  # "<" becomes "&lt;" etc., so nothing in the answer is read as HTML
    chips = _CITATION.sub(r'<span class="cite">\1</span>', safe)  # [S1] -> the chip S1
    chips = chips.replace("\n", "<br>")  # keeps the LLM's line breaks (lists, "Where:" lines); without this they are joined
    return f'<div class="answer">{chips}</div>'


def show_answer(text: str) -> None:
    """Draws a finished answer (used when the conversation is redrawn)."""
    html(format_answer(text))


#2 the answer appearing word by word

THINKING = '<div class="thinking"><span></span><span></span><span></span>Searching your course material</div>'


def stream_answer(question: str, cfg, history: list[dict] | None = None,
                  filters: Filters | None = None) -> AnswerResult | None:
    """Asks the engine and writes the answer as it arrives; returns the full result at the end.

    We don't use st.write_stream because it writes plain text: here every new word redraws the
    answer through format_answer, so the citation chips appear while the answer is being written.
    If the engine fails, it shows what went wrong and how to fix it, and returns None.
    """
    box = st.empty()  # one place on the page that we overwrite with the answer so far
    box.markdown(THINKING, unsafe_allow_html=True)  # 3 moving dots until the first word arrives
    text = ""  # the answer so far
    result = None  # the full result, the last item the engine sends
    try:
        for piece in answer_stream(question, history=history, filters=filters, cfg=cfg):
            if isinstance(piece, AnswerResult):  # the last item: the full result with the sources
                result = piece
            else:  # a piece of text
                text += piece  # adds it to the answer so far
                box.markdown(format_answer(text + "▌"), unsafe_allow_html=True)  # redraws it with a cursor at the end
        if result is None:  # the engine finished without sending the full result: a bug in the engine
            raise RuntimeError("the engine stopped without returning an AnswerResult")
    except Exception as error:  # anything the engine raises becomes a readable message, never a red traceback
        box.empty()  # removes the dots or the half-written answer
        error_notice(error)
        return None
    box.markdown(format_answer(result.answer_text), unsafe_allow_html=True)  # the final answer, without the cursor
    return result


#3 the notices

def explain_error(error: Exception) -> tuple[str, str]:
    """(what went wrong, how to fix it) in plain words, from the type and the text of the error."""
    message = str(error).lower()
    if isinstance(error, NotImplementedError):  # the real answer() isn't written yet
        return ("The answer engine isn't connected yet.",
                "Start the app with the fake engine: $env:USE_FAKE=1; streamlit run app.py")
    if isinstance(error, ConnectionError) or "connect" in message or "ollama" in message:
        return ("The local language model isn't running.",
                "Open the Ollama app, wait a few seconds and ask again.")
    if "not found" in message and "model" in message:  # Ollama is running but the model wasn't downloaded
        return ("The language model isn't installed on this laptop.",
                "Run: ollama pull llama3.1:8b (the model names are in configs/base.yaml).")
    if isinstance(error, FileNotFoundError) or "does not exist" in message or "collection" in message:
        return ("The knowledge base hasn't been built on this laptop.",
                "Run: python -m src.pipeline build --config configs/final.yaml")
    return ("Something went wrong while answering.", f"Details for the team: {type(error).__name__}: {error}")


def error_notice(error: Exception) -> None:
    """A red notice with what went wrong and the command that fixes it."""
    what, fix = explain_error(error)
    html(f"""
        <div class="notice warn">{icon("offline", 22)}
        <div><b>{html_lib.escape(what)}</b>{html_lib.escape(fix)}</div></div>
    """)


def refusal_notice() -> None:
    """Shown under a refusal, so the demo makes clear that refusing is on purpose."""
    html(f"""
        <div class="notice refuse">{icon("shield", 22)}
        <div><b>Not in your course material</b>
        {content.COMPANY_NAME} only answers from your syllabi and slides, so it says so instead of guessing.</div></div>
    """)


def no_sources_notice() -> None:
    """Shown under the LLM alone on the Compare page: an answer nobody can check."""
    html(f"""
        <div class="notice warn">{icon("page", 22)}
        <div><b>No sources</b>
        This answer comes from the model's memory. There is no slide or syllabus to check it against.</div></div>
    """)
