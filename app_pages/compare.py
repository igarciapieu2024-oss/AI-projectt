# Compare: the same question answered by the LLM alone and by our system, side by side.
# This page proves the point of the project live: the LLM alone invents, ours cites or refuses.
# The library that builds the page.
import streamlit as st

# The example questions.
import app_content as content
# How answers are drawn.
from app_components.answer_view import no_sources_notice, refusal_notice, stream_answer
# The settings.
from app_components.engine import config
# The source cards.
from app_components.sources import render_sources
# Our HTML helper.
from app_components.theme import html

# 1. Page header and the question
html(f'<h1 class="page-title">With and without {content.COMPANY_NAME}</h1>'
     '<div class="page-sub">The same local model answers the same question twice: once from its memory, once from your course material.</div>')

# The controls in one white bar.
with st.container(key="askbar"):
    # In one row.
    pick_col, type_col, button_col = st.columns([2, 2, 1], vertical_alignment="bottom")
    # One of the example questions.
    picked = pick_col.selectbox("Pick a question", content.EXAMPLE_QUESTIONS, index=0)
    # Optional: any other question.
    typed = type_col.text_input("Or write your own")
    # The typed one wins if there is one.
    question = typed.strip() or picked
    # Starts both answers.
    go = button_col.button("Compare", type="primary", icon=":material/compare_arrows:")

# 2. The two answers side by side
# Each panel: (key for the CSS, config, title, subtitle); baselines/b0_llm_only = no search, no sources.
# The RAG panel runs the earlier 3B system (baselines/v1_2_small_model, about 14 s an answer), not the final
# 8B system of the Assistant page (about 30 s).
PANELS = [
    ("panel_bare", "baselines/b0_llm_only", "The model alone", "Answers from what it memorised in training"),
    ("panel_rag", "baselines/v1_2_small_model", content.COMPANY_NAME, "Answers only from your syllabi and slides"),
]
if go and question:
    for col, (key, cfg_name, title, sub) in zip(st.columns(2, gap="medium"), PANELS):
        # The key gives the panel its colour line (red or green).
        with col, st.container(key=key):
            html(f'<div class="panel-title">{title}</div><div class="panel-sub">{sub}</div>')
            # The answer appears word by word.
            result = stream_answer(question, cfg=config(cfg_name))
            # The engine failed: stream_answer already showed what to fix.
            if result is None:
                continue
            if result.refused:
                refusal_notice()
            # The LLM alone: nothing to check.
            elif not result.context_hits:
                no_sources_notice()
            render_sources(result.context_hits, result.answer_text)
            # How long it took.
            st.caption(f"Answered in {result.timings_ms.get('total', 0) / 1000:.1f} s")
