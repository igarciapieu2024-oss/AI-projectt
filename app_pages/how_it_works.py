"""How it works: the architecture and the key results (owner: Irene). Texts and numbers come from app_content.py."""
import html as html_lib  # escapes the texts before putting them in HTML
from pathlib import Path  # to check if the architecture image exists

import streamlit as st  # the library that builds the page

import app_content as content  # the results and the architecture image path
from app_components.engine import courses, knowledge_base_stats  # the course names and the counts
from app_components.theme import html  # our HTML helper

esc = html_lib.escape  # a short name, used below

#1 page header
html('<h1 class="page-title">How it works</h1>'
     '<div class="page-sub">Two halves: the course material is prepared once, then every question goes through five steps.</div>')

#2 what it knows: the size of the knowledge base, counted from the real files on this laptop
stats = knowledge_base_stats()
figures = [  # (number, what it is); "—" while a number isn't available yet
    (stats["courses"], "courses"),
    (stats["syllabi"], "syllabi"),
    (stats["slides"] or stats["decks"], "slides" if stats["slides"] else "slide decks"),  # pages if PyMuPDF could count them
    (stats["images"], "images read"),
]
cells = "".join(f'<div><b>{n:,}</b><span>{esc(label)}</span></div>' if n else f'<div><b>—</b><span>{esc(label)}</span></div>'
                for n, label in figures)
names = ", ".join(courses().values())  # the course names, as in the filter
html(f'<div class="kb">{cells}</div><div class="course-list">Courses: {esc(names)}.</div>')

#3 the two halves of the system, drawn as two lanes of boxes and arrows
# each lane: (title, when it runs, [(step, what it does)]); the last box of the second lane is the answer, in yellow
LANES = [
    ("Built once: the knowledge base", "Run once on the laptop, before the demo", [
        ("Syllabi and slides", "PDFs of every course"),
        ("Clean the text", "headers, footers, duplicates out"),
        ("Read the images", "OCR, classify, describe"),
        ("Cut into chunks", "one per section or slide"),
        ("Index", "ChromaDB and keyword index"),
    ]),
    ("For every question", "A few seconds, fully offline", [
        ("Route", "admin, content, across courses or out of scope"),
        ("Search", "by meaning and by keywords"),
        ("Rerank", "keeps the best 5"),
        ("Self-check", "refuses if the sources don't cover it"),
        ("Answer", "local LLM, every sentence cited"),
    ]),
]
for i, (title, when, nodes) in enumerate(LANES):
    boxes = []
    for j, (step, what) in enumerate(nodes):
        last = i == len(LANES) - 1 and j == len(nodes) - 1  # the very last box: the cited answer
        boxes.append(f'<div class="node{" end" if last else ""}"><b>{esc(step)}</b><span>{esc(what)}</span></div>')
    html(f"""
        <div class="lane"><div class="lane-title">{esc(title)}</div><div class="lane-when">{esc(when)}</div>
        <div class="flow">{'<span class="arrow">›</span>'.join(boxes)}</div></div>
    """)

#4 the architecture figure from Sergio, when it arrives
if Path(content.ARCHITECTURE_IMAGE).exists():
    st.image(content.ARCHITECTURE_IMAGE)

#5 the key results: LLM alone vs our system (Elena's final numbers; "XX" until Sun 15)
rows = "".join(f'<tr><td>{esc(m)}</td><td class="num">{esc(a)}</td><td class="num ours">{esc(b)}</td></tr>'
               for m, a, b in content.RESULTS)
html(f"""
    <h2 class="section-title">Results on 60 test questions</h2>
    <table class="results"><thead><tr><th>Measure</th><th>The model alone</th><th class="ours">{esc(content.COMPANY_NAME)}</th></tr></thead>
    <tbody>{rows}</tbody></table>
""")
st.caption("Measured once on a locked test set, with 95% confidence intervals in the technical documentation.")
