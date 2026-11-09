"""Home: the landing page (owner: Irene). All texts come from app_content.py, all styles from assets/style.css."""
import html as html_lib  # escapes the texts before putting them in HTML

import streamlit as st  # the library that builds the web app

import app_content as content  # name, pitch, features, survey numbers, steps
from app_components.engine import USING_FAKE, knowledge_base_stats  # fake-engine note and the real counts
from app_components.theme import html, icon, image_data_uri  # our HTML helper, the icons and the logo loader

esc = html_lib.escape  # a short name, used many times below

#1 the hero: the pitch on the left, the real chat screen of the app on a laptop on the right


def headline() -> str:
    """The headline with the words between <em> and </em> in blue; everything else is escaped."""
    parts = content.TAGLINE.replace("</em>", "<em>").split("<em>")  # even pieces: normal text, odd pieces: blue
    return "".join(f"<em>{esc(p)}</em>" if i % 2 else esc(p) for i, p in enumerate(parts))


def laptop() -> str:
    """The laptop: a small copy of the Assistant page answering the demo question, with the cited parts highlighted."""
    answer = []
    for i, (text, cite) in enumerate(content.HERO_DEMO["answer"]):
        if cite:  # a cited part: highlighted, one after the other (the delay grows with each part)
            answer.append(f'<span class="hl" style="--d:{0.6 + 0.8 * i:.1f}s">{esc(text)}</span><span class="cite">{cite}</span>')
        else:  # plain text, e.g. the final full stop
            answer.append(esc(text))
    slips = "".join(f'<div class="slip"><span class="cite">{label}</span><b>{esc(where)}</b>{esc(section)}</div>'
                    for label, where, section in content.HERO_DEMO["sources"])  # one small card per source
    menu = "".join(f'<div class="{"on" if name == "Assistant" else ""}">{name}</div>'
                   for name in ("Home", "Assistant", "Compare", "How it works"))  # the same pages as the real menu
    n = len(content.HERO_DEMO["sources"])  # how many sources the demo answer cites

    # the 3 documents floating next to the laptop: what the assistant reads
    docs = "".join(f'<div class="doc {cls}"><div class="doc-icon">{icon(ic, 16)}</div>{esc(name)}<i></i><i></i></div>'
                   for cls, ic, name in (("d1", "page", "Syllabus"), ("d2", "slides", "Lecture slides"), ("d3", "image", "Diagrams")))
    return f"""
        <div class="device">{docs}
        <div class="laptop-screen"><div class="mini-app">
        <div class="mini-side"><img src="{image_data_uri(content.LOGO_PATH)}" alt="">{menu}</div>
        <div class="mini-main">
        <div class="mini-q">{esc(content.HERO_DEMO["question"])}</div>
        <p class="mini-a">{''.join(answer)}</p>
        <div class="slips">{slips}</div>
        </div></div></div>
        <div class="laptop-base"></div>
        <div class="float-badge"><div class="check">{icon("check", 16)}</div>
        <div><b>Checked against {n} sources</b><span>Every claim is in the syllabus</span></div></div>
        </div>
    """


with st.container(key="hero"):  # the key gives the container a CSS class (.st-key-hero) so we can style it
    left, right = st.columns([1, 1.05], gap="large", vertical_alignment="center")  # text on the left, laptop on the right
    with left:
        html(f"""
            <div class="hero-pill">{icon("offline", 16)}{esc(content.HERO_PILL)}</div>
            <h1 class="hero-title">{headline()}</h1>
            <p class="hero-pitch">{esc(content.PITCH)}</p>
        """)
        if st.button("Ask a question", type="primary", key="cta_top", icon=":material/chat_bubble:"):  # straight to the chat
            st.switch_page("app_pages/assistant.py")
    with right:
        html(laptop())

#2 what it does: 4 cards, then the strip with the 3 guarantees (with the real size of the knowledge base)
features = "".join(f'<div class="feature"><div class="feature-icon">{icon(ic, 26)}</div><b>{esc(t)}</b><span>{esc(d)}</span></div>'
                   for ic, t, d in content.FEATURES)
stats = knowledge_base_stats()  # counted from the real files on this laptop
counts = {k: f"{v:,}" if v else "—" for k, v in stats.items()}  # 2622 -> "2,622"; missing -> "—"
trust = "".join(f'<div>{icon(ic, 30)}<div><b>{esc(t)}</b><span>{esc(d.format(**counts))}</span></div></div>'
                for ic, t, d in content.TRUST)
html(f"""
    <h2 class="section-title center">An assistant for your courses</h2>
    <div class="features">{features}</div>
    <div class="trust">{trust}</div>
""")

#3 the problem, with the survey numbers ("XX" until Francesca sends them)
problem = "".join(f'<div class="stat"><div class="stat-n">{esc(n)}</div><div class="stat-l">{esc(t)}</div></div>'
                  for n, t in content.PROBLEM_STATS)
html(f'<h2 class="section-title">Course information is hard to find</h2><div class="stats">{problem}</div>')

#4 how it works, in 4 numbered steps (they are a real sequence, so the numbers mean something)
steps = "".join(f'<div class="step"><div class="step-n">{i}</div><b>{esc(t)}</b><span>{esc(d)}</span></div>'
                for i, (t, d) in enumerate(content.STEPS, start=1))
html(f'<h2 class="section-title">From question to cited answer</h2><div class="steps">{steps}</div>')

#5 the closing band: a second button, for whoever scrolled to the end
with st.container(key="closing"):
    text_col, button_col = st.columns([3, 1], vertical_alignment="center")
    with text_col:
        html(f'<div class="closing-title">{esc(content.CLOSING)}</div>')
    with button_col:
        if st.button("Ask a question", type="primary", key="cta_bottom"):
            st.switch_page("app_pages/assistant.py")

#6 footer
note = " Demo mode: answers come from the fake engine." if USING_FAKE else ""  # so nobody thinks the fake answers are real
html(f'<div class="footer">{esc(content.FOOTER)}{note}</div>')
