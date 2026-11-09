"""The "behind the scenes" panel: what the system did to answer (owner: Irene)."""
from __future__ import annotations  # lets us write types like "dict[str, float]" in any Python version

import html as html_lib  # escapes the rewritten question before putting it in HTML

import streamlit as st  # the library that builds the page

from app_components.theme import html  # our HTML helper
from src.schemas import AnswerResult  # the agreed format of an answer

# plain-language names for the router categories, so the panel is readable in the demo
ROUTE_NAMES = {
    "course_admin": ("Course admin", "searched the syllabi"),
    "course_content": ("Course content", "searched the slides"),
    "cross_course": ("Across courses", "used the course table"),
    "out_of_scope": ("Out of scope", "refused without searching"),
    "none": ("No retrieval", "the LLM alone"),
}
# the steps of one answer, in order, with a readable name and a colour for the time bar
STEPS = [
    ("rewrite", "Rewrite follow-up", "#8A97AB"),
    ("route", "Route", "#5B6080"),
    ("retrieve", "Search", "#2B6FD6"),
    ("rerank", "Rerank", "#6E56CF"),
    ("generate", "Write answer", "#241C63"),
]


def _time_bar(timings_ms: dict[str, float]) -> str:
    """One horizontal bar split by step, each part as wide as the time it took (shows the LLM takes most of it)."""
    parts = [(name, label, color, timings_ms.get(name, 0) / 1000) for name, label, color in STEPS]
    parts = [p for p in parts if p[3] > 0]  # skips the steps that didn't run (e.g. no follow-up)
    total = sum(p[3] for p in parts) or 1  # avoids dividing by zero
    bar = "".join(f'<div style="flex:{s / total:.4f};background:{c}" title="{l}: {s:.2f} s">'
                  f'{f"{s:.1f} s" if s / total > 0.12 else ""}</div>' for _, l, c, s in parts)  # time written only on wide parts
    legend = "".join(f'<span><i style="background:{c}"></i>{l} {s:.2f} s</span>' for _, l, c, s in parts)
    return f'<div class="timebar">{bar}</div><div class="timelegend">{legend}</div>'


def render_behind(result: AnswerResult) -> None:
    """Route, rewritten question, filters, time per step and the score of each source."""
    with st.expander("Behind the scenes", expanded=False):  # closed by default, opened in the demo
        name, action = ROUTE_NAMES.get(result.route.label, (result.route.label, ""))
        confidence = f'<span class="pill">{result.route.prob:.0%} confident</span>' if result.route.prob else ""
        rewritten = ""  # only shown for follow-up questions
        if result.rewritten_question != result.question:
            rewritten = f'<span class="pill">Read as: {html_lib.escape(result.rewritten_question)}</span>'
        filters = ""  # only shown if a course or session filter was applied
        if result.filters_used.course_code or result.filters_used.sessions:
            filters = (f'<span class="pill">Course {result.filters_used.course_code or "any"}, '
                       f'sessions {result.filters_used.sessions or "any"}</span>')
        total = result.timings_ms.get("total", 0) / 1000
        html(f"""
            <div class="route"><b>{name}</b><span>{action}</span>{confidence}{rewritten}{filters}</div>
            <div class="src-head">Time per step, {total:.1f} s in total</div>
            {_time_bar(result.timings_ms)}
        """)

        if result.context_hits:  # the score of each source at each search stage, as an HTML table (no pandas needed)
            html(f"""
                <div class="src-head">Score of each source at each search stage</div>
                <table class="scores"><thead><tr>
                <th>Source</th><th>From</th><th>Meaning search</th><th>Keyword search</th><th>Reranker</th>
                </tr></thead><tbody>{''.join(_score_row(h) for h in result.context_hits)}</tbody></table>
            """)


def _number(value) -> str:
    """A score with 3 decimals, or a dash when that search stage didn't score this source."""
    return "—" if value is None else f"{value:.3f}"


def _score_row(hit) -> str:
    """One row of the scores table: the source, where it comes from and its 3 scores."""
    return (f'<tr><td><span class="tag label">{hit.label}</span></td>'
            f'<td>{html_lib.escape(hit.chunk.outline_path)}</td>'
            f'<td class="n">{_number(hit.score_dense)}</td>'  # ChromaDB similarity (meaning)
            f'<td class="n">{_number(hit.score_bm25)}</td>'  # BM25 (keywords)
            f'<td class="n">{_number(hit.score_rerank)}</td></tr>')  # the final score that chose the top 5
