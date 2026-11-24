# The Streamlit app: the entry point with the 3 pages (Home, Assistant, Compare).
#   streamlit run app.py                 the real system: needs Ollama open and the knowledge base built
#   USE_FAKE=1 streamlit run app.py      the fake engine (src/generation/fake.py), to work on the pages without models
#   Windows PowerShell:                  $env:USE_FAKE=1; streamlit run app.py
import logging

# The library that builds the web app.
import streamlit as st

# The company name and the logo files.
import app_content as content
from app_components.engine import USING_FAKE, warm_up_models
# Our CSS (colours, fonts, cards).
from app_components.theme import apply_theme

logger = logging.getLogger(__name__)

# 1. Browser tab and page width
# The tab shows the logo and the name; "wide" uses the whole screen width, better for the projector.
st.set_page_config(page_title=f"{content.COMPANY_NAME}, {content.SUBTITLE.lower()}", page_icon=content.LOGO_MARK, layout="wide")

# Load the models and the indexes once, when the app starts (src/warmup.py), so the first question is not slower
# than the rest; the fake engine doesn't use them. If the warm-up fails, the error is written in the terminal and
# the app still opens.
if not USING_FAKE:
    try:
        warm_up_models()
    except Exception:
        logger.exception("Model warm-up failed; the app will continue starting.")

# 2. The brand
# Loads assets/style.css before any page is drawn.
apply_theme()
# Logo with the name, at the left of the top bar.
st.logo(content.LOGO_PATH, size="large")

# 3. The 3 pages; each one is a file in app_pages/, and the menu at the top is built from this list
# The icons are Google's Material icons, which come inside Streamlit, so they work offline.
pages = [
    # The landing page opens first.
    st.Page("app_pages/home.py", title="Home", icon=":material/home:", default=True),
    # The chat, the technical core.
    st.Page("app_pages/assistant.py", title="Assistant", icon=":material/forum:"),
    # Our system vs the LLM alone.
    st.Page("app_pages/compare.py", title="Compare", icon=":material/compare:"),
]
# Menu in a bar at the top, like a normal web app; runs the chosen page.
st.navigation(pages, position="top").run()
