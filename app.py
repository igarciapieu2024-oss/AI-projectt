"""Streamlit chat (owner: Irene).

Run with the fake engine (works today):   USE_FAKE=1 streamlit run app.py
Run with the real engine (from Wed 11):   streamlit run app.py
Windows PowerShell:  $env:USE_FAKE=1; streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

from src.schemas import AnswerResult

if os.environ.get("USE_FAKE") == "1":
    from src.generation.fake import answer_stream
else:
    from src.generation.answer import answer_stream

st.set_page_config(page_title="Course assistant", page_icon="📚")
st.title("Course assistant")
st.caption("Answers only from the official syllabi and slides, offline.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask about your courses")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        history = st.session_state.messages[:-1]
        result: AnswerResult | None = None

        def text_only():
            global result
            for piece in answer_stream(question, history=history):
                if isinstance(piece, AnswerResult):
                    result = piece
                else:
                    yield piece

        st.write_stream(text_only())
        if result is not None:
            with st.expander("Sources"):
                for hit in result.context_hits:
                    st.markdown(f"**[{hit.label}]** {hit.chunk.outline_path}")
                    st.caption(hit.chunk.text)
            st.session_state.messages.append({"role": "assistant", "content": result.answer_text})
