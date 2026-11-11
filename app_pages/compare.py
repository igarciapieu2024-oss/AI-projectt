"""Compare: the same question answered by the LLM alone and by our system (owner: Irene).

This page proves the point of the project live: the LLM alone invents, ours cites or refuses.
"""
import streamlit as st  # the library that builds the page

import app_content as content  # the example questions
from app_components.answer_view import no_sources_notice, refusal_notice, stream_answer  # how answers are drawn
from app_components.engine import config  # the settings
from app_components.sources import render_sources  # the source cards
from app_components.theme import html  # our HTML helper

#1 page header and the question
html(f'<h1 class="page-title">With and without {content.COMPANY_NAME}</h1>'
     '<div class="page-sub">The same local model answers the same question twice: once from its memory, once from your course material.</div>')

with st.container(key="askbar"):  # the controls in one white bar
    pick_col, type_col, button_col = st.columns([2, 2, 1], vertical_alignment="bottom")  # in one row
    picked = pick_col.selectbox("Pick a question", content.EXAMPLE_QUESTIONS, index=0)  # one of the example questions
    typed = type_col.text_input("Or write your own")  # optional: any other question
    question = typed.strip() or picked  # the typed one wins if there is one
    go = button_col.button("Compare", type="primary", icon=":material/compare_arrows:")  # starts both answers

#2 the two answers side by side
# each panel: (key for the CSS, config, title, subtitle); b0_no_rag = no search, no sources
PANELS = [
    ("panel_bare", "b0_no_rag", "The model alone", "Answers from what it memorised in training"),
    ("panel_rag", "final", content.COMPANY_NAME, "Answers only from your syllabi and slides"),
]
if go and question:
    for col, (key, cfg_name, title, sub) in zip(st.columns(2, gap="medium"), PANELS):
        with col, st.container(key=key):  # the key gives the panel its colour line (red or green)
            html(f'<div class="panel-title">{title}</div><div class="panel-sub">{sub}</div>')
            result = stream_answer(question, cfg=config(cfg_name))  # the answer appears word by word
            if result is None:  # the engine failed: stream_answer already showed what to fix
                continue
            if result.refused:
                refusal_notice()
            elif not result.context_hits:  # the LLM alone: nothing to check
                no_sources_notice()
            render_sources(result.context_hits, result.answer_text)
            st.caption(f"Answered in {result.timings_ms.get('total', 0) / 1000:.1f} s")  # how long it took
