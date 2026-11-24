# Every text and number the app shows. The pages read them from here, so changing a text, a number or a result
# means editing only this file (and the images in assets/), never the pages.

# Company (from the business case).
COMPANY_NAME = "SyllabusAI"
# Used in the browser tab.
SUBTITLE = "The course assistant of your university"
# The big headline of the landing page; the words between <em> and </em> are shown in blue.
# The mission, from the business plan.
TAGLINE = "A <em>correct, cited</em> answer about every course, in seconds."
PITCH = ("SyllabusAI answers students' questions using only the official material the university has loaded: "
         "the syllabi and the session slides, including the charts and diagrams inside them. Every answer cites the "
         "exact slide or syllabus section, and when the material doesn't cover a question, it says so instead of guessing.")
# The small label above the headline.
HERO_PILL = "Official course material only, on the university's own servers"
# The logo with the name, in one line: the top bar and the laptop in the hero.
LOGO_PATH = "assets/logo.png"
# The logo without the name: browser tab, chat avatar, welcome screen.
LOGO_MARK = "assets/logo_mark.png"
# The original logo, name under the icon: for slides and documents.
LOGO_FULL = "assets/logo_full.png"

# The answer shown in the hero: a real question with the real answer from the Management Control syllabus.
HERO_DEMO = {
    "question": "How much is the final exam worth in Management Control?",
    # (text, citation or None); the parts with a citation get the highlighter.
    "answer": [
        ("The final exam is worth 40% of the grade, and you need at least 4.5 in it to pass", "S1"),
        (". Missing the 80% attendance rule means re-enrolling the next year", "S2"),
        (".", None),
    ],
    "sources": [("S1", "Management Control · Syllabus", "Evaluation criteria"),
                ("S2", "Management Control · Syllabus", "Re-sit / re-take policy")],
}

# The problem, in numbers (from the business plan: BBA office interview, 12 Nov 2026; Digital Education Council, 2024).
PROBLEM_TITLE = "The answers are already written. Students still ask."
# (number, what it means); 3 numbers shown as big figures on the landing page.
PROBLEM_STATS = [
    ("26 h", "a week the BBA office spends answering questions already in official documents"),
    ("90%", "of those questions could be answered from the syllabus or the student guide"),
    ("86%", "of students already use AI in their studies, mostly general tools that don't know their courses"),
]
PROBLEM_SOURCE = "Sources: BBA Programme Office, IE University (Nov 2026); Digital Education Council Global AI Student Survey (2024)."

# How it works, in 4 steps for students (landing page).
# (title, text).
STEPS = [
    ("Ask in your own words", "Type a question about any of your courses, or follow up on the last one."),
    ("It finds the passages", "It searches every syllabus and slide, including the text inside diagrams."),
    ("It checks it has enough", "If your material doesn't cover the question, it tells you instead of guessing."),
    ("You get a cited answer", "Every sentence points to the syllabus section or slide it comes from."),
]

# What it does (4 cards on the landing page); the icon names are in app_components/theme.py.
# (icon, title, text); the differences with general chatbots, from the business plan.
FEATURES = [
    ("cite", "Cited, or it says so", "Every answer points to the exact slide or syllabus section. If the material doesn't cover a question, it says so."),
    ("layers", "All your courses at once", "One place for every course, including questions that compare them, like which final exam weighs most."),
    ("image", "Reads charts and diagrams", "Text inside slide images is read and described, which matters in visual subjects like Macroeconomics."),
    ("campus", "Stays inside the university", "Runs on the university's own computers: no student data leaves it, no outside provider."),
]
# The strip under the cards: (icon, title, text); {courses}, {syllabi}, {slides} are filled with the real counts.
TRUST = [
    ("page", "Loaded once by the university", "Official syllabi and slides, for every student"),
    ("shield", "Student data stays inside", "Nothing is sent to an outside server"),
    ("slides", "This prototype", "{courses} courses, {syllabi} syllabi, {slides} slides"),
]

# Who uses it and who buys it (business plan, sections 1.4, 1.6 and 1.7).
AUDIENCE_TITLE = "Built for the whole programme"
# (icon, who, what they get).
AUDIENCE = [
    ("users", "Students", "A correct, cited answer about any of their courses, from a laptop or a phone, in English or Spanish."),
    ("cite", "Professors", "Fewer repeated questions about exam formats and grading, and students who study the right model."),
    ("campus", "Programme office", "Time back from questions that are already answered in the syllabus or the student guide."),
]
MISSION = "A correct, cited answer about every course, in seconds."
VISION = "The trusted knowledge layer of every university."
FOR_UNIVERSITIES = ("Universities license SyllabusAI per year, by number of enrolled students, with a one-time setup "
                    "to load their courses. The syllabi and slides are reloaded every semester.")
# The dark band at the end of the landing page.
CLOSING = "Try it with a question from one of the 7 courses of the pilot."
FOOTER = "SyllabusAI, a student project at IE University, 2026."

# Example questions: the starter cards of the chat and the Compare page.
# One per route of the router, so the demo shows all 4 behaviours; all of them are answered in the real PDFs.
EXAMPLE_QUESTIONS = [
    # Course admin: Management Control syllabus, evaluation criteria.
    "What is the final exam worth in Management Control?",
    # Course content: Strategies, session 4 slides (Industry analysis).
    "What are Porter's five forces?",
    # Across courses: the course table.
    "Which of my courses need a minimum grade in the final exam?",
    # Out of scope: the system refuses instead of guessing.
    "When is the end-of-year party?",
]
# The icon of each example question card (Google's Material icons, they come inside Streamlit, so offline).
EXAMPLE_ICONS = [":material/grading:", ":material/hub:", ":material/compare_arrows:", ":material/celebration:"]

# Course names as the students know them, when the manifest spells one differently (only changes what the app shows).
COURSE_DISPLAY_NAMES = {"ob": "Organizational Behaviour", "ai_ml": "Machine Learning Foundations"}

# Placeholder courses for the filter, used until data/manifest.csv has rows.
PLACEHOLDER_COURSES = {"c01": "Course 1", "c02": "Course 2"}
