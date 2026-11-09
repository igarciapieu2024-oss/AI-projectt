"""Every text and number the app shows that will change later (owner: Irene).

The pages read from here, so when the real name, logo, survey numbers or results arrive,
we only edit this file (and drop the images into assets/), never the pages.
"XX" marks the values we still don't have.
"""

# company (from the business case)
COMPANY_NAME = "SyllabusAI"
SUBTITLE = "Your offline course assistant"  # used in the browser tab
# the big headline of the landing page; the words between <em> and </em> are shown in blue
TAGLINE = "Ask questions about your <em>syllabi</em> and <em>lecture slides</em>."
PITCH = ("SyllabusAI answers from your official syllabi and session slides, including the text inside their "
         "diagrams. Every answer cites the page it comes from, and if your material doesn't cover a question, "
         "it tells you instead of guessing.")
HERO_PILL = "Offline, private, built from your courses"  # the small label above the headline
LOGO_PATH = "assets/logo.png"  # the logo with the name, in one line: the top bar and the laptop in the hero
LOGO_MARK = "assets/logo_mark.png"  # the logo without the name: browser tab, chat avatar, welcome screen
LOGO_FULL = "assets/logo_full.png"  # the original logo, name under the icon: for slides and documents

# the answer shown in the hero: a real question with the real answer from the Management Control syllabus
HERO_DEMO = {
    "question": "How much is the final exam worth in Management Control?",
    # (text, citation or None); the parts with a citation get the highlighter
    "answer": [
        ("The final exam is worth 40% of the grade, and you need at least 4.5 in it to pass", "S1"),
        (". Missing the 80% attendance rule means re-enrolling the next year", "S2"),
        (".", None),
    ],
    "sources": [("S1", "Management Control · Syllabus", "Evaluation criteria"),
                ("S2", "Management Control · Syllabus", "Re-sit / re-take policy")],
}

# the problem, in numbers (from the student survey)
PROBLEM_STATS = [  # (number, what it means); 3 numbers shown as big figures on the landing page
    ("XX%", "of students struggle to find course information in syllabi and slides"),
    ("XX%", "have asked ChatGPT about a course and got a wrong answer"),
    ("XX min", "spent per week looking for course information"),
]

# how it works, in 4 steps for students (landing page)
STEPS = [  # (title, text)
    ("Ask in your own words", "Type a question about any of your courses, or follow up on the last one."),
    ("It finds the passages", "It searches every syllabus and slide, including the text inside diagrams."),
    ("It checks it has enough", "If your material doesn't cover the question, it tells you instead of guessing."),
    ("You get a cited answer", "Every sentence points to the syllabus section or slide it comes from."),
]

# what it does (4 cards on the landing page); the icon names are in app_components/theme.py
FEATURES = [  # (icon, title, text)
    ("cite", "Cited answers", "Every answer points to the syllabus section or slide it comes from, with a picture of the slide."),
    ("image", "Reads slides and diagrams", "Text inside images is read and described, so an answer can come from a chart or a diagram."),
    ("shield", "Says when it doesn't know", "If your course material doesn't cover a question, it tells you instead of making something up."),
    ("offline", "Offline and private", "Runs on the laptop: no internet, and no question or document ever leaves it."),
]
# the strip under the cards: (icon, title, text); {courses}, {syllabi}, {slides} are filled with the real counts
TRUST = [
    ("offline", "100% offline", "No internet needed"),
    ("shield", "Your data stays here", "Nothing is sent to a server"),
    ("layers", "Built from your courses", "{courses} courses, {syllabi} syllabi, {slides} slides"),
]
CLOSING = "Try it with a question from your own courses."  # the dark band at the end of the landing page
FOOTER = "SyllabusAI, a student project at IE University, 2026."

# example questions: the starter cards of the chat and the Compare page (Elena confirms the 4 that work best on Sat 14)
# one per route of the router, so the demo shows all 4 behaviours; all of them are answered in the real PDFs
EXAMPLE_QUESTIONS = [
    "What is the final exam worth in Management Control?",  # course admin: Management Control syllabus, evaluation criteria
    "What are Porter's five forces?",  # course content: Strategies, session 4 slides (Industry analysis)
    "Which of my courses need a minimum grade in the final exam?",  # across courses: the course table
    "When is the end-of-year party?",  # out of scope: the system refuses instead of guessing
]
# the icon of each example question card (Google's Material icons, they come inside Streamlit, so offline)
EXAMPLE_ICONS = [":material/grading:", ":material/hub:", ":material/compare_arrows:", ":material/celebration:"]

# course names as the students know them, when the manifest spells one differently (only changes what the app shows)
COURSE_DISPLAY_NAMES = {"ob": "Organizational Behaviour", "ai_ml": "Machine Learning Foundations"}

# placeholder courses for the filter, used until data/manifest.csv has rows
PLACEHOLDER_COURSES = {"c01": "Course 1", "c02": "Course 2"}

# results for the How-it-works page (Elena's final numbers, Sun 15)
RESULTS = [  # (metric, bare LLM, our system)
    ("Correct answers", "XX%", "XX%"),
    ("Answers with invented facts", "XX%", "XX%"),
    ("Right source found in the top 5", "—", "XX%"),
    ("Unanswerable questions refused", "XX%", "XX%"),
]
ARCHITECTURE_IMAGE = "assets/architecture.png"  # Sergio's architecture figure (Sat 14)
