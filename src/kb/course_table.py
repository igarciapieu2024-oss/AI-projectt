# Build the course table: one row per course with the facts students ask about most, read from the syllabus sections
# (data/interim/syllabus_sections.jsonl, written by parse_syllabi.py): final exam weight, evaluation components and
# their weights, minimum attendance, re-sit policy, professor and number of sessions. Each row also keeps the ids of
# the sections every fact came from.
# The output is data/processed/course_table.csv, one row per course, in the CourseRow format of src/schemas.py. We use
# it in three places: syllabus_text.py turns each row into the summary chunk of its course, table_qa.py answers the
# questions that compare courses directly from it, and answer.py uses its course names to find which course a question
# names (for the course filter).
from __future__ import annotations

# json stores the components inside one CSV cell; logging writes warnings; re finds weights and names in the text;
# defaultdict groups the sections by course.
import json
import logging
import re
from collections import defaultdict

# The folders of the project, the formats of a course row and a syllabus section, and the readers and writers.
from src.config import PATHS
from src.schemas import CourseRow, SyllabusSection
from src.utils.io import read_csv, read_jsonl, write_csv

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Patterns

# A weight: a number of up to 3 digits, with an optional decimal (point or comma), followed by "%", "percent" or
# "percentage". The lookarounds stop it from matching inside a longer number or word.
_WEIGHT = re.compile(
    r"(?<![\w.])(?P<weight>\d{1,3}(?:[.,]\d+)?)\s*(?:%|percent(?:age)?)(?![A-Za-z])",
    re.IGNORECASE,
)
# One spelling for each component that the syllabi write in different ways. The key is the name in lower case without
# spaces or symbols, so "FinalExam", "Final exam" and "final examination" all become "Final Exam", and we don't count
# the same component twice.
_COMPONENT_ALIASES = {
    "finalexam": "Final Exam",
    "finalexamination": "Final Exam",
    "intermediatetest": "Intermediate Test",
    "intermediatetests": "Intermediate Tests",
    "individualwork": "Individual Work",
    "midtermexam": "Midterm",
    "midtermexams": "Midterms",
    "midterm": "Midterm",
    "classtest": "Class Test",
    "classtests": "Class Tests",
    "classparticipation": "Class Participation",
}
# Words that name an evaluation component. In the running text, a percentage only counts as a component when the words
# before it contain one of these, so we ignore the other percentages of the syllabus.
_COMPONENT_TERMS = re.compile(
    r"\b(exams?|examinations?|tests?|quizzes|assignments?|participation|projects?|"
    r"presentations?|coursework|assessments?|midterms?|finals?|practicals?|"
    r"exercises?|cases?|papers?|reports?|work)\b",
    re.IGNORECASE,
)
# Phrases that show a percentage is a rule and not a component (absences, minimum grades, special cases), so we skip
# those sentences even if they contain a component word.
_NON_COMPONENT_PHRASES = re.compile(
    r"\b(?:personal|work reasons|will have|weight of|missing|absence|absent|"
    r"grade of|in case|because)\b",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# Evaluation components

# The number of a weight match as a float; a decimal comma becomes a point.
def _weight_value(match: re.Match[str]) -> float:
    return float(match.group("weight").replace(",", "."))


# Clean the name of a component so the same component always gets the same name.
def _clean_component_name(value: str) -> str:
    # Remove a list marker at the start, such as "A." or "1)".
    value = re.sub(r"^\s*(?:[A-Z]|\d+)[.)]\s*", "", value.strip())
    # Remove a bracket that was opened but not closed before the weight.
    value = re.sub(r"\s*\([^)]*$", "", value)
    # Remove spaces, colons and dashes at the end, and full stops at both ends.
    value = re.sub(r"[\s:–—-]+$", "", value).strip(" .")
    # Remove a leading "the".
    value = re.sub(r"^(?:the\s+)?", "", value, flags=re.IGNORECASE)
    # Remove the verb that links the name to its weight.
    value = re.sub(
        r"\s+(?:is|are|worth|represents?|accounts?\s+for)\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    # Separate words glued by the PDF reader.
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    # We use the common spelling when the name is a known component; otherwise every word gets a capital letter.
    compact = re.sub(r"[^a-z]", "", value.casefold())
    if compact in _COMPONENT_ALIASES:
        return _COMPONENT_ALIASES[compact]
    return " ".join(word.capitalize() for word in value.split())


# Build one component {"name", "weight"}, or None when the name has no letters or the weight isn't between 0 and 100
# (then it isn't a real component).
def _component(name: str, weight: float) -> dict[str, float | str] | None:
    # Clean the name first.
    name = _clean_component_name(name)
    if not name or not re.search(r"[A-Za-z]", name) or not 0 <= weight <= 100:
        return None
    return {"name": name, "weight": weight}


# Read the components from a Markdown table of the syllabus: in each row, the first weight we find and the name in
# front of it.
def _table_components(table: str) -> list[dict[str, float | str]]:
    # The components found in this table.
    components = []
    for line in table.splitlines():
        # Only table rows (they start with "|").
        if not line.strip().startswith("|"):
            continue
        # Split the row into cells; "\|" is a "|" written inside a cell.
        cells = [cell.strip().replace(r"\|", "|") for cell in line.strip().strip("|").split("|")]
        # Skip the separator line under the header ("| --- | --- |").
        if not cells or all(re.fullmatch(r"[:\-\s]+", cell or "-") for cell in cells):
            continue
        # Every weight in the row, with the cell it's in; rows without a weight aren't components.
        matches = [(index, match) for index, cell in enumerate(cells) for match in _WEIGHT.finditer(cell)]
        if not matches:
            continue
        # The first weight of the row is the component's weight.
        weight_index, weight_match = matches[0]
        # The name is the text before the weight in the same cell ("Final Exam 40%")
        name = cells[weight_index][:weight_match.start()]
        # or, if that is empty, the nearest filled cell to the left ("| Final Exam | 40% |").
        if not name.strip():
            name = next(
                (cell for cell in reversed(cells[:weight_index]) if cell.strip()),
                "",
            )
        # Build the component; one that isn't valid is left out.
        item = _component(name, _weight_value(weight_match))
        if item:
            components.append(item)
    return components


# Read the components from running text, for syllabi that describe the evaluation in sentences or lists instead of a
# table. With require_component_term, a percentage only counts when the words before it name a component (see
# _COMPONENT_TERMS).
def _text_components(text: str, *, require_component_term: bool) -> list[dict[str, float | str]]:
    # The components found in this text.
    components = []
    # Split the text into pieces at line breaks, semicolons, dot leaders and bullets, so each piece holds at most one
    # component.
    segments = re.split(r"\n|;|…|\.{2,}|•", text)
    # A piece without a weight is kept as the possible name of the next piece, for lists where the name and the weight
    # are on different lines.
    previous_label = ""
    for segment in segments:
        # Look for a weight in this piece.
        match = _WEIGHT.search(segment)
        if not match:
            previous_label = segment.strip()
            continue
        # The name is the text before the weight, or the previous line if there's no text before it.
        name = segment[:match.start()]
        if not name.strip() and previous_label:
            name = previous_label
        previous_label = ""
        # When a component word is required, a name without one is skipped.
        if require_component_term and not _COMPONENT_TERMS.search(name):
            continue
        # We skip rules about absences or grades, and long sentences (more than 6 words), which are explanations and
        # not component names.
        if _NON_COMPONENT_PHRASES.search(name) or len(name.split()) > 6:
            continue
        # Build the component; one that isn't valid is left out.
        item = _component(name, _weight_value(match))
        if item:
            components.append(item)
    return components


# A key to compare two components by name: lower case and letters only. "Participation" and "Class Participation"
# count as the same component.
def _component_key(component: dict[str, float | str]) -> str:
    # Lower case and letters only.
    key = re.sub(r"[^a-z]", "", str(component["name"]).casefold())
    return "participation" if key in {"participation", "classparticipation"} else key


# True if the list already has the same component with the same weight, so a component written twice in a syllabus is
# only counted once.
    
def _contains_component(
    components: list[dict[str, float | str]],
    candidate: dict[str, float | str],
) -> bool:
    return any(
        _component_key(item) == _component_key(candidate)
        and abs(float(item["weight"]) - float(candidate["weight"])) < 0.01
        for item in components
    )


# ----------------------------------------------------------------------
# Other facts of the course

# Each function below returns the value it found and the id of the section it came from, so the row can list its
# evidence. They only look in the sections that parse_syllabi.py labelled with the right name.

# Minimum attendance: a percentage within a few words of "attendance", before or after it ("attendance below 80%",
# "80% attendance"), in the ATTENDANCE or RESIT sections.
    
def _attendance_value(sections: list[SyllabusSection]) -> tuple[float | None, str | None]:
    # A percentage up to 60 characters after "attendance", or "attendance" up to 40 characters after a percentage,
    # inside the same sentence.
    attendance_pattern = re.compile(
        r"\battendance\b[^\n.;]{0,60}?(?P<after>\d{1,3}(?:[.,]\d+)?)\s*%"
        r"|(?P<before>\d{1,3}(?:[.,]\d+)?)\s*%[^\n.;]{0,40}\battendance\b",
        re.IGNORECASE,
    )
    # Look only in the ATTENDANCE and RESIT sections.
    for section in sections:
        if section.section_name not in {"ATTENDANCE", "RESIT"}:
            continue
        # The text of the section together with its tables.
        content = "\n".join([section.text, *section.tables])
        match = attendance_pattern.search(content)
        if match:
            # The number is in one of the two groups, depending on which side of "attendance" it was.
            value = match.group("after") or match.group("before")
            return float(value.replace(",", ".")), section.section_id
    # No syllabus section states it.
    return None, None


# Re-sit policy: the first sentences (at most 3 and about 300 characters) of the RESIT section, without its title
# line, so the summary chunk stays short.
    
def _retake_policy(sections: list[SyllabusSection]) -> tuple[str, str | None]:
    # The title line of the section ("RE-SIT / RE-TAKE POLICY"), which we remove.
    title_pattern = re.compile(
        r"^\s*(?:re[\s-]?sit|re[\s-]?take|retake)"
        r"(?:\s*/\s*(?:re[\s-]?sit|re[\s-]?take|retake))?\s+policy\s*$",
        re.IGNORECASE,
    )
    # Look only in the RESIT section.
    for section in sections:
        if section.section_name != "RESIT":
            continue
        # The lines of the section without table bars, without empty lines and without the title.
        lines = [line.strip(" |") for line in section.text.splitlines()]
        lines = [line for line in lines if line and not title_pattern.fullmatch(line)]
        # Join them into one text and split it into sentences.
        content = re.sub(r"\s+", " ", " ".join(lines)).strip()
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", content)
            if sentence.strip()
        ]
        # We add sentences until we have 3 or the next one would go past 300 characters.
        selected = []
        for sentence in sentences[:3]:
            if selected and sum(len(value) + 1 for value in selected) + len(sentence) > 300:
                break
            selected.append(sentence)
        if selected:
            return " ".join(selected), section.section_id
    return "", None


# Professor: the name after "Professor:" or "Instructor:" in the PROFESSOR section, without the e-mail.
def _professor(sections: list[SyllabusSection]) -> tuple[str, str | None]:
    # A line that starts with "Professor:" or "Instructor:", and the pattern of an e-mail address.
    professor_pattern = re.compile(
        r"^\s*(?:professor|instructor)\s*:\s*(.+?)\s*$",
        re.IGNORECASE,
    )
    email_pattern = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
    # Look only in the PROFESSOR section, line by line.
    for section in sections:
        if section.section_name != "PROFESSOR":
            continue
        for line in section.text.splitlines():
            match = professor_pattern.match(line)
            if match:
                # The name without the e-mail and without punctuation left at the ends.
                name = email_pattern.sub("", match.group(1)).strip(" \t,;()")
                if name:
                    return name, section.section_id
    return "", None


# Number of sessions: the highest session number in the PROGRAM section. "SESSIONS 29 - 30" counts as 30, so a
# programme that ends with a range isn't undercounted.
    
def _session_count(sections: list[SyllabusSection]) -> tuple[int | None, str | None]:
    # "SESSION 4" or a range such as "SESSIONS 29 - 30".
    session_pattern = re.compile(
        r"\bSESSIONS?\s+(\d+)(?:\s*(?:-|–|—|to)\s*(\d+))?\b",
        re.IGNORECASE,
    )
    # The highest number found so far, and the section it was in.
    highest: int | None = None
    source_id: str | None = None
    for section in sections:
        if section.section_name != "PROGRAM":
            continue
        # The text of the section together with its tables.
        content = "\n".join([section.text, *section.tables])
        for match in session_pattern.finditer(content):
            # The numbers of this match (one, or two for a range), and the highest of them.
            values = [int(value) for value in match.groups() if value is not None]
            candidate = max(values)
            if highest is None or candidate > highest:
                highest = candidate
                source_id = section.section_id
    return highest, source_id


# ----------------------------------------------------------------------
# One row per course

# Build the row of one course from its syllabus sections.
def _course_row(
    course_code: str,
    manifest_row: dict[str, str],
    sections: list[SyllabusSection],
) -> CourseRow:
    # We look for the components twice in the EVALUATION sections: in their tables and in their text.
    evaluation_sections = [section for section in sections if section.section_name == "EVALUATION"]
    # First the tables: every component found in a table of an EVALUATION section, without repeats, and the section it
    # came from.
    table_components: list[dict[str, float | str]] = []
    table_sources: list[str] = []
    for section in evaluation_sections:
        for table in section.tables:
            for item in _table_components(table):
                if not _contains_component(table_components, item):
                    table_components.append(item)
                    table_sources.append(section.section_id)

    # What the weights of the table components add up to.
    table_total = sum(float(item["weight"]) for item in table_components)
    # Then the text of the same sections, only percentages whose name has a component word.
    text_components = []
    text_sources = []
    for section in evaluation_sections:
        for item in _text_components(section.text, require_component_term=True):
            if not _contains_component(text_components, item):
                text_components.append(item)
                text_sources.append(section.section_id)
    text_total = sum(float(item["weight"]) for item in text_components)

    # Now we choose which list to keep. A list whose weights add up to about 100% (±1) is taken as complete.
    text_is_complete = bool(text_components) and abs(text_total - 100) <= 1
    # A complete table wins, because the text often names the same parts differently. The AI/ML table says "FinalExam
    # 35%" while its text says "Quiz (final) [35%]", which hid the final exam weight.
    if table_components and abs(table_total - 100) <= 1:
        components = table_components
        component_sources = table_sources
    # Otherwise, a complete list from the text.
    elif text_is_complete:
        components = text_components
        component_sources = text_sources
    else:
        # Neither is complete. Some tables were cut by the PDF reader, and their last component and weight ended up in
        # a section labelled OTHER: we add those if the total then reaches 100%.
        components = table_components
        component_sources = table_sources
        for section in sections:
            if section.section_name != "OTHER":
                continue
            continuation = _text_components(section.text, require_component_term=True)
            # Try adding the components of this OTHER section that we don't have yet.
            combined = list(components)
            added = []
            for item in continuation:
                if not _contains_component(combined, item):
                    combined.append(item)
                    added.append(item)
            # If with them the total reaches about 100%, we keep the combined list and stop looking.
            combined_total = sum(float(item["weight"]) for item in combined)
            if added and abs(combined_total - 100) <= 1:
                components = combined
                component_sources.extend([section.section_id] * len(added))
                break
        # No table components at all: we keep what the text gave, even if it's incomplete.
        if not components:
            components = text_components
            component_sources = text_sources

    # The final exam weight is the weight of the component called "final exam" or "final examination"; None if the
    # course has no final exam.
    final_exam_weight = next(
        (
            float(item["weight"])
            for item in components
            if re.search(r"\bfinal\s+(?:exam|examination)\b", str(item["name"]), re.IGNORECASE)
        ),
        None,
    )

    # The other facts, each with the section it was found in.
    attendance_limit_pct, attendance_source = _attendance_value(sections)
    retake_policy, retake_source = _retake_policy(sections)
    professor, professor_source = _professor(sections)
    n_sessions, sessions_source = _session_count(sections)

    # The evidence: the ids of every section a fact came from, in the order of the syllabus, joined with ";". The
    # summary chunk built from this row lists them in `covers`, so the evaluation counts it as a correct source when
    # the expected source is one of these sections.
    evidence_ids = set(component_sources)
    for source_id in (attendance_source, retake_source, professor_source, sessions_source):
        if source_id:
            evidence_ids.add(source_id)
    # Keep the order of the syllabus.
    evidence_section_ids = ";".join(
        section.section_id for section in sections if section.section_id in evidence_ids
    )

    # A warning, not an error, when the weights don't add up to about 100%: we still write the row, and the message
    # shows which syllabus to check by hand.
    if components and abs(sum(float(item["weight"]) for item in components) - 100) > 1:
        logger.warning(
            "%s: evaluation components sum to %.1f%%, not approximately 100%%",
            course_code,
            sum(float(item["weight"]) for item in components),
        )

    # We store the components as JSON text, because a CSV cell holds one value.
    return CourseRow(
        course_code=course_code,
        course_name=manifest_row.get("course_name", "").strip(),
        final_exam_weight=final_exam_weight,
        components=json.dumps(components, ensure_ascii=False),
        attendance_limit_pct=attendance_limit_pct,
        retake_policy=retake_policy,
        professor=professor,
        n_sessions=n_sessions,
        language=manifest_row.get("language", "").strip(),
        evidence_section_ids=evidence_section_ids,
    )


# Build one row per course and write data/processed/course_table.csv. It returns the number of rows. The build
# pipeline (src/pipeline.py) calls it after parse_syllabi and before the chunking, because the chunking makes a
# summary chunk from each row.
def build_course_table(cfg) -> int:
    # We don't need the settings: all the paths come from src.config.PATHS.
    del cfg
    # One manifest row per course, to get its name and its language. We prefer the syllabus row, because a course has
    # one syllabus and many slide decks.
    manifest_rows = read_csv(PATHS["manifest"])
    courses: dict[str, dict[str, str]] = {}
    # Go through the manifest and keep one row per course.
    for row in manifest_rows:
        course_code = row.get("course_code", "").strip()
        if not course_code:
            raise ValueError("manifest.csv contains a row without course_code")
        if course_code not in courses or row.get("doc_type", "").strip().lower() == "syllabus":
            courses[course_code] = row

    # A manifest without courses is a mistake.
    if not courses:
        raise ValueError(f"No courses found in manifest: {PATHS['manifest']}")

    # Group the syllabus sections by course.
    section_records = read_jsonl(PATHS["syllabus_sections"], SyllabusSection)
    sections_by_course: dict[str, list[SyllabusSection]] = defaultdict(list)
    for section in section_records:
        sections_by_course[section.course_code].append(section)

    # Build the row of every course from its sections, and write the CSV with the columns of CourseRow.
    rows = [
        _course_row(course_code, manifest_row, sections_by_course[course_code])
        for course_code, manifest_row in courses.items()
    ]
    return write_csv(PATHS["course_table"], rows, CourseRow.field_names())
