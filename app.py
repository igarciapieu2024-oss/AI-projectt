"""The Streamlit app: entry point with the 4 pages (owner: Irene).

Run with the fake engine (works today):   USE_FAKE=1 streamlit run app.py
Windows PowerShell:                         $env:USE_FAKE=1; streamlit run app.py
Run with the real engine (from Wed 11):    streamlit run app.py
"""
import streamlit as st  # the library that builds the web app

import app_content as content  # the company name and the logo files
from app_components.engine import USING_FAKE, warm_up_models
from app_components.theme import apply_theme  # our CSS (colours, fonts, cards)

#1 browser tab and page width
# the tab shows the logo and the name; "wide" uses the whole screen width, better for the projector
st.set_page_config(page_title=f"{content.COMPANY_NAME}, {content.SUBTITLE.lower()}", page_icon=content.LOGO_MARK, layout="wide")

# Load local retrieval models before the first assistant question. The fake app
# does not need the real knowledge-base models.
if not USING_FAKE:
    warm_up_models()

#2 the brand
apply_theme()  # loads assets/style.css before any page is drawn
st.logo(content.LOGO_PATH, size="large")  # logo with the name, at the left of the top bar

#3 the 4 pages; each one is a file in app_pages/, and the menu at the top is built from this list
# the icons are Google's Material icons, which come inside Streamlit, so they work offline
pages = [
    st.Page("app_pages/home.py", title="Home", icon=":material/home:", default=True),  # the landing page opens first
    st.Page("app_pages/assistant.py", title="Assistant", icon=":material/forum:"),  # the chat, the technical core
    st.Page("app_pages/compare.py", title="Compare", icon=":material/compare:"),  # our system vs the LLM alone
    st.Page("app_pages/how_it_works.py", title="How it works", icon=":material/account_tree:"),  # architecture and results
]
st.navigation(pages, position="top").run()  # menu in a bar at the top, like a normal web app; runs the chosen page
