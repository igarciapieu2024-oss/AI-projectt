# How an answer is written on screen: word by word, with the [S1] citations as yellow chips, plus the notices for
# refusals, errors and answers without sources. Used by the Assistant and Compare pages.
# Lets us write types like "list[dict] | None" in any Python version.
from __future__ import annotations

# Escapes < and > in the answer, so the LLM's text can't break our HTML.
import html as html_lib
# Finds the [S1], [S2]... citations.
import re

# The library that builds the page.
import streamlit as st

# The product name.
import app_content as content
# The fake or the real engine.
from app_components.engine import answer_stream
# Our HTML helper and the icons.
from app_components.theme import html, icon
# The agreed formats of an answer and of the filters.
from src.schemas import AnswerResult, Filters

# A citation in the answer text, e.g. [S3].
_CITATION = re.compile(r"\[(S\d+)\]")


# 1. The answer text with citation chips

# The answer as HTML: the text escaped, and every [S1] turned into a yellow chip.
def format_answer(text: str) -> str:
    # "<" becomes "&lt;" etc., so nothing in the answer is read as HTML.
    safe = html_lib.escape(text)
    # [S1] -> the chip S1.
    chips = _CITATION.sub(r'<span class="cite">\1</span>', safe)
    # Keeps the LLM's line breaks (lists, "Where:" lines); without this they are joined.
    chips = chips.replace("\n", "<br>")
    return f'<div class="answer">{chips}</div>'


# Draws a finished answer (used when the conversation is redrawn).
def show_answer(text: str) -> None:
    html(format_answer(text))


# 2. The answer appearing word by word

THINKING = '<div class="thinking"><span></span><span></span><span></span>Searching your course material</div>'


# Asks the engine and writes the answer as it arrives; returns the full result at the end.
# st.write_stream isn't used because it writes plain text: here every new word redraws the answer through
# format_answer, so the citation chips appear while the answer is being written.
# If the engine fails, it shows what went wrong and how to fix it, and returns None.
def stream_answer(question: str, cfg, history: list[dict] | None = None,
                  filters: Filters | None = None) -> AnswerResult | None:
    # One place on the page that we overwrite with the answer so far.
    box = st.empty()
    # 3 moving dots until the first word arrives.
    box.markdown(THINKING, unsafe_allow_html=True)
    # The answer so far.
    text = ""
    # The full result, the last item the engine sends.
    result = None
    try:
        for piece in answer_stream(question, history=history, filters=filters, cfg=cfg):
            # The last item: the full result with the sources.
            if isinstance(piece, AnswerResult):
                result = piece
            # A piece of text.
            else:
                # Adds it to the answer so far.
                text += piece
                # Redraws it with a cursor at the end.
                box.markdown(format_answer(text + "▌"), unsafe_allow_html=True)
        # The engine finished without sending the full result: a bug in the engine.
        if result is None:
            raise RuntimeError("the engine stopped without returning an AnswerResult")
    # Anything the engine raises becomes a readable message, never a red traceback.
    except Exception as error:
        # Removes the dots or the half-written answer.
        box.empty()
        error_notice(error)
        return None
    # The final answer, without the cursor.
    box.markdown(format_answer(result.answer_text), unsafe_allow_html=True)
    return result


# 3. The notices

# (what went wrong, how to fix it) in plain words, from the type and the text of the error.
def explain_error(error: Exception) -> tuple[str, str]:
    message = str(error).lower()
    # A part of the engine that isn't written.
    if isinstance(error, NotImplementedError):
        return ("The answer engine isn't connected yet.",
                "Start the app with the fake engine: $env:USE_FAKE=1; streamlit run app.py")
    if isinstance(error, ConnectionError) or "connect" in message or "ollama" in message:
        return ("The local language model isn't running.",
                "Open the Ollama app, wait a few seconds and ask again.")
    # Ollama is running but the model wasn't downloaded.
    if "not found" in message and "model" in message:
        return ("The language model isn't installed on this laptop.",
                "Run ollama pull with the model configured as generation.llm_model in configs/base.yaml.")
    if isinstance(error, FileNotFoundError) or "does not exist" in message or "collection" in message:
        return ("The knowledge base hasn't been built on this laptop.",
                "Run: python -m src.pipeline build --config configs/final_system.yaml")
    return ("Something went wrong while answering.", f"Details for the team: {type(error).__name__}: {error}")


# A red notice with what went wrong and the command that fixes it.
def error_notice(error: Exception) -> None:
    what, fix = explain_error(error)
    html(f"""
        <div class="notice warn">{icon("offline", 22)}
        <div><b>{html_lib.escape(what)}</b>{html_lib.escape(fix)}</div></div>
    """)


# Shown under a refusal, so the demo makes clear that refusing is on purpose.
def refusal_notice() -> None:
    html(f"""
        <div class="notice refuse">{icon("shield", 22)}
        <div><b>Not in your course material</b>
        {content.COMPANY_NAME} only answers from your syllabi and slides, so it says so instead of guessing.</div></div>
    """)


# Shown under the LLM alone on the Compare page: an answer nobody can check.
def no_sources_notice() -> None:
    html(f"""
        <div class="notice warn">{icon("page", 22)}
        <div><b>No sources</b>
        This answer comes from the model's memory. There is no slide or syllabus to check it against.</div></div>
    """)
