# Assistant: the chat, the technical core of the app. It runs the system chosen in app_components/engine.py
# (final_system by default) and shows each answer word by word, with its sources and the behind-the-scenes panel.
# Escapes the texts before putting them in HTML.
import html as html_lib

# The library that builds the page.
import streamlit as st

# The example questions and the logo.
import app_content as content
# How answers are drawn (and errors).
from app_components.answer_view import refusal_notice, show_answer, stream_answer
# The "behind the scenes" panel.
from app_components.behind import render_behind
# The settings, the system the app runs and the course list.
from app_components.engine import MAIN_CONFIG, config, courses
# The source cards.
from app_components.sources import render_sources
# Our HTML helper and the logo loader.
from app_components.theme import html, image_data_uri
# The agreed answer format and the filters format.
from src.schemas import AnswerResult, Filters

# 1. A narrower page, like a chat app: the conversation in the middle of the screen
html("""<style>
    [data-testid="stMainBlockContainer"], [data-testid="stBottomBlockContainer"] { max-width: 880px; }
</style>""")

# The chat history lives in the session, so it survives each click.
if "messages" not in st.session_state:
    # Each message: {"role", "content", "result"}.
    st.session_state.messages = []

# 2. The toolbar: course filter, behind-the-scenes switch and "New chat", in one white bar
with st.container(key="toolbar"):
    course_col, behind_col, new_col = st.columns([2.2, 1.6, 1], vertical_alignment="center")
    # course_code -> course_name, from the manifest.
    all_courses = courses()
    choice = course_col.selectbox("Course", ["All courses", *all_courses.values()],
                                  # No label: the options say what it is.
                                  label_visibility="collapsed")
    course_code = None if choice == "All courses" else next(c for c, n in all_courses.items() if n == choice)
    # Shows the panel under each answer.
    show_behind = behind_col.toggle("Behind the scenes", value=False)
    if new_col.button("New chat", icon=":material/add:", disabled=not st.session_state.messages):
        # Starts a new conversation.
        st.session_state.messages = []
        # Redraws the page empty.
        st.rerun()

# 3. The conversation so far


# Everything under an answer: the refusal note, the source cards and the panel.
def show_result(result: AnswerResult) -> None:
    # Refusals in their own style, so the demo shows the system knows when to stop.
    if result.refused:
        refusal_notice()
    render_sources(result.context_hits, result.answer_text)
    if show_behind:
        render_behind(result)


# The key lets the CSS give the answers their white card.
with st.container(key="chatlog"):
    # Redraws the whole conversation on every click (that's how Streamlit works).
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=content.LOGO_MARK if msg["role"] == "assistant" else None):
            # Assistant messages: the answer with chips, its sources and the panel.
            if msg.get("result"):
                show_answer(msg["content"])
                show_result(msg["result"])
            # A turn where the engine failed: a short reminder, the details were shown then.
            elif msg.get("error"):
                st.caption("No answer: the engine wasn't available. Ask again once it's fixed.")
            # User messages: plain text.
            else:
                st.markdown(msg["content"])

# Becomes the question text if an example is clicked.
clicked = None

# 4. Empty chat: a welcome and the example questions as cards, so the first click of the demo is easy
if not st.session_state.messages:
    # Which course the answers come from.
    where = "all your courses" if course_code is None else html_lib.escape(choice)
    html(f"""
        <div class="welcome"><img src="{image_data_uri(content.LOGO_MARK)}" alt="">
        <div class="welcome-title">What do you want to know?</div>
        <div class="welcome-sub">Answers come only from the syllabi and slides of {where}, with the source of every claim.</div></div>
    """)
    # Styled as cards in assets/style.css.
    with st.container(key="starters"):
        # 2 per row.
        cols = st.columns(2)
        for i, q in enumerate(content.EXAMPLE_QUESTIONS):
            if cols[i % 2].button(q, key=f"start_{i}", icon=content.EXAMPLE_ICONS[i % len(content.EXAMPLE_ICONS)]):
                clicked = q

# 5. After an answer: the other example questions as small chips, so the demo can go on with one click
elif st.session_state.messages[-1]["role"] == "assistant":
    # Questions already asked.
    asked = {m["content"] for m in st.session_state.messages if m["role"] == "user"}
    # The ones still to try.
    left = [q for q in content.EXAMPLE_QUESTIONS if q not in asked]
    if left:
        with st.container(key="suggest"):
            # A new key after every answer, so the chips start unselected each time.
            clicked = st.pills("Try next", left, key=f"suggest_{len(st.session_state.messages)}")

# 6. A new question

# Typed or clicked.
question = st.chat_input("Ask about your courses…") or clicked
if question:
    # Saves the question.
    st.session_state.messages.append({"role": "user", "content": question})
    # Redraws the page: the question appears in the chat (and the welcome goes away).
    st.rerun()

# The last message is a question without an answer yet: answer it now.
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    question = st.session_state.messages[-1]["content"]
    # The history sent to the engine: only the text of previous turns, so it can understand follow-ups.
    history = [{"role": m["role"], "content": m["content"]}
               # Failed turns are left out.
               for m in st.session_state.messages[:-1] if not m.get("error")]
    # The course chosen in the toolbar.
    filters = Filters(course_code=course_code)
    with st.container(key="chatlog_new"), st.chat_message("assistant", avatar=content.LOGO_MARK):
        # Word by word, with chips.
        result = stream_answer(question, cfg=config(MAIN_CONFIG), history=history, filters=filters)
        # None = the engine failed; stream_answer already showed what to fix.
        if result is not None:
            show_result(result)
    # Saved as a failed turn, so the page doesn't retry on every click; asking again retries.
    if result is None:
        st.session_state.messages.append({"role": "assistant", "content": "", "error": question})
        # Keeps the error message on screen.
        st.stop()
    st.session_state.messages.append({"role": "assistant", "content": result.answer_text, "result": result})
    # Redraws once more, so the suggested next questions appear under the answer.
    st.rerun()
