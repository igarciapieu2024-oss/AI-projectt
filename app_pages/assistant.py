"""Assistant: the chat, the technical core of the app (owner: Irene)."""
import html as html_lib  # escapes the texts before putting them in HTML

import streamlit as st  # the library that builds the page

import app_content as content  # the example questions and the logo
from app_components.answer_view import refusal_notice, show_answer, stream_answer  # how answers are drawn (and errors)
from app_components.behind import render_behind  # the "behind the scenes" panel
from app_components.engine import config, courses  # the settings and the course list
from app_components.sources import render_sources  # the source cards
from app_components.theme import html, image_data_uri  # our HTML helper and the logo loader
from src.schemas import AnswerResult, Filters  # the agreed answer format and the filters format

#1 a narrower page, like a chat app: the conversation in the middle of the screen
html("""<style>
    [data-testid="stMainBlockContainer"], [data-testid="stBottomBlockContainer"] { max-width: 880px; }
</style>""")

if "messages" not in st.session_state:  # the chat history lives in the session, so it survives each click
    st.session_state.messages = []  # each message: {"role", "content", "result"}

#2 the toolbar: course filter, behind-the-scenes switch and "New chat", in one white bar
with st.container(key="toolbar"):
    course_col, behind_col, new_col = st.columns([2.2, 1.6, 1], vertical_alignment="center")
    all_courses = courses()  # course_code -> course_name, from the manifest
    choice = course_col.selectbox("Course", ["All courses", *all_courses.values()],
                                  label_visibility="collapsed")  # no label: the options say what it is
    course_code = None if choice == "All courses" else next(c for c, n in all_courses.items() if n == choice)
    show_behind = behind_col.toggle("Behind the scenes", value=False)  # shows the panel under each answer
    if new_col.button("New chat", icon=":material/add:", disabled=not st.session_state.messages):
        st.session_state.messages = []  # starts a new conversation
        st.rerun()  # redraws the page empty

#3 the conversation so far


def show_result(result: AnswerResult) -> None:
    """Everything under an answer: the refusal note, the source cards and the panel."""
    if result.refused:  # refusals in their own style, so the demo shows the system knows when to stop
        refusal_notice()
    render_sources(result.context_hits, result.answer_text)
    if show_behind:
        render_behind(result)


with st.container(key="chatlog"):  # the key lets the CSS give the answers their white card
    for msg in st.session_state.messages:  # redraws the whole conversation on every click (that's how Streamlit works)
        with st.chat_message(msg["role"], avatar=content.LOGO_MARK if msg["role"] == "assistant" else None):
            if msg.get("result"):  # assistant messages: the answer with chips, its sources and the panel
                show_answer(msg["content"])
                show_result(msg["result"])
            elif msg.get("error"):  # a turn where the engine failed: a short reminder, the details were shown then
                st.caption("No answer: the engine wasn't available. Ask again once it's fixed.")
            else:  # user messages: plain text
                st.markdown(msg["content"])

clicked = None  # becomes the question text if an example is clicked

#4 empty chat: a welcome and the example questions as cards, so the first click of the demo is easy
if not st.session_state.messages:
    where = "all your courses" if course_code is None else html_lib.escape(choice)  # which course the answers come from
    html(f"""
        <div class="welcome"><img src="{image_data_uri(content.LOGO_MARK)}" alt="">
        <div class="welcome-title">What do you want to know?</div>
        <div class="welcome-sub">Answers come only from the syllabi and slides of {where}, with the source of every claim.</div></div>
    """)
    with st.container(key="starters"):  # styled as cards in assets/style.css
        cols = st.columns(2)  # 2 per row
        for i, q in enumerate(content.EXAMPLE_QUESTIONS):
            if cols[i % 2].button(q, key=f"start_{i}", icon=content.EXAMPLE_ICONS[i % len(content.EXAMPLE_ICONS)]):
                clicked = q

#5 after an answer: the other example questions as small chips, so the demo can go on with one click
elif st.session_state.messages[-1]["role"] == "assistant":
    asked = {m["content"] for m in st.session_state.messages if m["role"] == "user"}  # questions already asked
    left = [q for q in content.EXAMPLE_QUESTIONS if q not in asked]  # the ones still to try
    if left:
        with st.container(key="suggest"):
            # a new key after every answer, so the chips start unselected each time
            clicked = st.pills("Try next", left, key=f"suggest_{len(st.session_state.messages)}")

#6 a new question

question = st.chat_input("Ask about your courses…") or clicked  # typed or clicked
if question:
    st.session_state.messages.append({"role": "user", "content": question})  # saves the question
    st.rerun()  # redraws the page: the question appears in the chat (and the welcome goes away)

# the last message is a question without an answer yet: answer it now
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    question = st.session_state.messages[-1]["content"]
    # the history sent to the engine: only the text of previous turns, so it can understand follow-ups
    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.messages[:-1] if not m.get("error")]  # failed turns are left out
    filters = Filters(course_code=course_code)  # the course chosen in the toolbar
    with st.container(key="chatlog_new"), st.chat_message("assistant", avatar=content.LOGO_MARK):
        result = stream_answer(question, cfg=config("final"), history=history, filters=filters)  # word by word, with chips
        if result is not None:  # None = the engine failed; stream_answer already showed what to fix
            show_result(result)
    if result is None:  # saved as a failed turn, so the page doesn't retry on every click; asking again retries
        st.session_state.messages.append({"role": "assistant", "content": "", "error": question})
        st.stop()  # keeps the error message on screen
    st.session_state.messages.append({"role": "assistant", "content": result.answer_text, "result": result})
    st.rerun()  # redraws once more, so the suggested next questions appear under the answer
