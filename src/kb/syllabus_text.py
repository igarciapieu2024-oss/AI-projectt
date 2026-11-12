"""Clean text for the knowledge base: readable syllabus tables and one summary chunk per course (owner: Irene).

Why this exists:
    PyMuPDF reads the syllabus tables with the spaces and decimal points missing, e.g.
    "| FinalExam | 40% | ...budgetsandevaluatehow... | Aminimumgradeof . 45isrequired... |" or "| Lectures | 133% . | 200hours . |".
    That text is hard to match for the search and hard to read for the LLM. Here each table row becomes one short line,
    e.g. "- Final Exam: 40%. A minimum grade of 4.5 is required on the final exam to pass the subject.",
    and every course gets a clean summary chunk built from the course table ("final exam 40%, attendance 80%, ...").

Only the text of the chunks changes: section_ids and slide_ids stay the same, so the evaluation's gold_ids still match.
"""
from __future__ import annotations  # lets us write types like "list[str] | None" in any Python version

import json  # the course table stores the evaluation components as a JSON list
import re  # finds the glued words, the lost decimals and the percentage cells

try:  # splits glued words ("Thefinalexam" -> "The final exam") with an English word list; optional
    import wordninja
except ImportError:  # without it the tables are still cleaned, only the glued words stay glued
    wordninja = None

from src.config import PATHS  # the shared folder paths of the project
from src.schemas import Chunk, Parent  # the agreed formats of one chunk and one parent
from src.utils.io import read_csv, read_jsonl  # course table, manifest, syllabus sections

# 1. runs of letters from this length are checked for glued words; real words ("Discussions") come back unchanged
GLUED_MIN_LETTERS = 8
# 2. short words that are fine inside a split; any other part shorter than 3 letters means the split is wrong
#    (e.g. "regularization" -> "regular iz ation" is rejected and the word stays as it is)
# the one split the word list gets wrong in our syllabi ("will coverall course content")
SPLIT_FIXES = {"coverall": "cover all"}
SHORT_WORDS = {"a", "i", "an", "as", "at", "be", "by", "do", "in", "is", "it", "of", "on", "or", "to", "we", "up", "no"}
# 3. a table cell that is a weight: "40%", "40 %", "133% ."
PERCENT_CELL = re.compile(r"^\d{1,4}\s?%(\s?\.)?$")
# 4. the decimal point the PDF loses: "133% ." is 13.3 %, "200hours ." is 20.0 hours, "of . 45 is" is 4.5
LOST_DECIMAL_PERCENT = re.compile(r"\b(\d+)(\d)\s?%\s?\.")
LOST_DECIMAL_HOURS = re.compile(r"\b(\d+)(\d)\s?hours\s?\.")
LOST_DECIMAL_LOOSE = re.compile(r"(?<=\s)\.\s(\d)(\d)(?!\d)")


def _split_glued(word: str) -> str:
    """One glued word -> its words ("Individualwork" -> "Individual work"), or the word unchanged if unsure."""
    if wordninja is None or len(word) < GLUED_MIN_LETTERS:
        return word
    parts = wordninja.split(word)
    if len(parts) < 2 or any(len(p) < 3 and p.lower() not in SHORT_WORDS for p in parts):
        return word  # a real long word ("Macroeconomics") or a doubtful split: keep it
    return " ".join(SPLIT_FIXES.get(p.lower(), p) for p in parts)


def clean_cell(text: str) -> str:
    """Make one table cell readable: CamelCase apart, glued words apart, decimals back, extra spaces out."""
    text = LOST_DECIMAL_PERCENT.sub(r"\1.\2%", text)
    text = LOST_DECIMAL_HOURS.sub(r"\1.\2 hours", text)
    text = LOST_DECIMAL_LOOSE.sub(r"\1.\2", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)  # "FinalExam" -> "Final Exam"
    text = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", text)  # "session14anditwill" -> "session 14 anditwill"
    text = re.sub(r"[A-Za-z]+", lambda m: _split_glued(m.group(0)), text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)  # "subject ." -> "subject."
    text = re.sub(r"(?<=\w)\s*-\s+(?=\w)", " ", text)  # line-break hyphens "Understand the different - types"
    return re.sub(r"\s+", " ", text).strip(" |")


def _rows(table: str) -> list[list[str]]:
    """The cells of each row of a markdown table, without the "| --- |" separator lines."""
    rows = []
    for line in table.splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|- "):
            continue
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def table_to_text(table: str) -> str:
    """One markdown table from a syllabus -> short readable lines.

    5. evaluation tables (criteria | percentage | learning objectives | comments): "- Final Exam: 40%. <comment>"
       The learning-objectives column is dropped: it repeats the course objectives in every row and buries the weights.
    6. workload tables (activity | weighting | hours): "- Lectures: 13.3% of the time, 20.0 hours"
    7. any other table: its cells joined with " | ", cleaned
    """
    rows = _rows(table)
    if not rows:
        return ""
    lines = []
    for cells in rows:
        cleaned = [clean_cell(c) for c in cells]
        if len(cleaned) >= 2 and PERCENT_CELL.match(cells[1]):
            name, weight = cleaned[0].rstrip(" ,"), cleaned[1]
            if len(cleaned) >= 4:  # evaluation row: keep the comment, drop the learning objectives
                comment = cleaned[3]
                lines.append(f"- {name}: {weight}." + (f" {comment}" if comment else ""))
            elif len(cleaned) == 3:  # workload row
                lines.append(f"- {name}: {weight} of the time, {cleaned[2]}")
            else:
                lines.append(f"- {name}: {weight}")
        elif any(c.lower() in {"criteria", "learning activity"} for c in cleaned):
            continue  # header row: the lines above already say what each value is
        elif any(cleaned):
            lines.append(" | ".join(c for c in cleaned if c))
    return "\n".join(lines)


def section_text(section: dict) -> str:
    """The text of one syllabus section for the chunks: its body plus its tables written as readable lines."""
    text = fix_broken_hyphens(section.get("text", ""))
    tables = [table_to_text(t) for t in section.get("tables", [])]
    tables = [t for t in tables if t]
    if tables:
        text += "\n\n" + "\n\n".join(tables)
    return text


# 15. sentence ends: a full stop / ? / ! followed by a space and a capital, a digit, a quote or a bracket
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(\[-])")
# 16. where a new session starts in the syllabus programme: "SESSION 4", "SESSIONS 9-10-11", "SESSIONS 29 - 30"
SESSION_START = re.compile(r"(?=\bSESSIONS?\s+\d)")
SESSION_NUMBERS = re.compile(r"\bSESSIONS?\s+((?:\d+\s*(?:-|,|&|and)?\s*)+)")


def _fit_words(text: str, size: int) -> list[str]:
    """A single sentence longer than a chunk: cut it at spaces, never inside a word."""
    words, pieces, current = text.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > size:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    return pieces + ([current] if current else [])


def split_into_chunks(text: str, size: int, overlap: int, by_session: bool = False) -> list[str]:
    """Cut a text into chunks of at most `size` characters at sentence ends, never in the middle of a word.

    17. the old cut fell wherever 800 characters ended, so chunks started like "rganizations . Elaborate...";
        now whole sentences (and whole table lines) are packed together until the next one does not fit
    18. `overlap`: the last sentences of a chunk (up to that many characters) are repeated at the start of the
        next one, so an idea split between two chunks is still complete in one of them
    19. by_session (the syllabus programme): the units are whole sessions instead of sentences, with no overlap,
        so each chunk holds complete sessions and can be tagged with their numbers
    """
    units = []
    if by_session:
        blocks = [b for b in SESSION_START.split(text or "") if b.strip()]
        overlap = 0
    else:
        # the PDF breaks sentences at the end of each printed line, so lines are joined back into paragraphs;
        # a line starting with "- " (a bullet or a table line) or after an empty line starts a new block
        blocks, current_block = [], []
        for line in (text or "").splitlines():
            if not line.strip() or line.lstrip().startswith("- "):
                if current_block:
                    blocks.append(" ".join(current_block))
                current_block = [line] if line.strip() else []
            else:
                current_block.append(line)
        if current_block:
            blocks.append(" ".join(current_block))
    for block in blocks:
        block = " ".join(block.split())
        # a whole session, or a table/bullet line ("- Final Exam: 40%. <comment>"), stays one unit when it fits
        whole = by_session or (block.startswith("- ") and len(block) <= size)
        sentences = [block] if whole else SENTENCE_END.split(block)
        for sentence in sentences:
            units.extend([sentence] if len(sentence) <= size else _fit_words(sentence, size))
    chunks, current = [], []
    for unit in units:
        if current and len(" ".join(current + [unit])) > size:
            chunks.append(" ".join(current))
            carried = []  # the overlap: last sentences of this chunk, up to `overlap` characters
            for previous in reversed(current):
                if len(" ".join([previous] + carried)) > overlap:
                    break
                carried.insert(0, previous)
            current = carried if len(" ".join(carried + [unit])) <= size else []
        current.append(unit)
    if current:
        chunks.append(" ".join(current))
    return chunks


def session_numbers(piece: str, previous: list[int]) -> list[int]:
    """The session numbers a programme chunk covers ("SESSIONS 9-10-11" -> [9, 10, 11]).

    20. a chunk that does not start with "SESSION" continues the last session of the chunk before, so it keeps it
    """
    numbers = []
    for match in SESSION_NUMBERS.finditer(piece):
        numbers += [int(n) for n in re.findall(r"\d+", match.group(1))]
    if previous and not re.match(r"\s*SESSIONS?\s+\d", piece):
        numbers = [previous[-1]] + numbers
    return sorted(set(n for n in numbers if 0 < n < 100))


def fix_broken_hyphens(text: str) -> str:
    """21. words split at a line end by the PDF: "(re-\\nsits)" / "(re- sits)" -> "(re-sits)" (not "short- and long-term")."""
    return re.sub(r"(?<=[a-z])-[ \t]*\n?[ \t]*(?=[a-z]{2,}\b)(?!and\b|or\b|to\b)", "-", text)


def course_names() -> dict[str, str]:
    """course_code -> course name, from data/manifest.csv ("mgmt_control" -> "Management Control")."""
    path = PATHS["manifest"]
    if not path.is_file():
        return {}
    return {row["course_code"]: row["course_name"] for row in read_csv(path)}


def section_heading(section: dict) -> str:
    """The title a syllabus section really has, for its outline path.

    10. 36 of the 92 sections are labelled OTHER even though they have a clear heading ("SUBJECT DESCRIPTION",
        "LEARNING OBJECTIVES", "BEHAVIOR RULES"...), so the outline "Syllabus › OTHER" told the search nothing.
        The first line is used when it looks like a heading (short, no colon, mostly capitals); the very first
        section is the course header (course name, credits, number of sessions), so it is called "Course details".
    """
    first = section.get("text", "").strip().split("\n", 1)[0].strip().rstrip(":")
    letters = [ch for ch in first if ch.isalpha()]
    looks_like_heading = 0 < len(first) <= 60 and ":" not in first and letters and (
        sum(ch.isupper() for ch in letters) / len(letters) > 0.6
    )
    if section.get("section_id", "").endswith("_sec01"):
        return "Course details"
    if looks_like_heading:
        return first.title().replace("Ai ", "AI ").replace("Re-Sit", "Re-sit").replace("Re-Take", "Re-take")
    return section.get("section_name", "Section").replace("_", " ").title().replace("Ai ", "AI ")


def with_course(course_name: str, outline_path: str, text: str) -> str:
    """The text that is embedded, searched and reranked: course name + outline + text.

    11. 2,102 of the 3,371 chunks never said which course they belong to (a slide titled "Introduction" could be
        any course), so a question that names the course could not tell them apart without the course filter.
    """
    prefix = f"{course_name} › {outline_path}" if course_name else outline_path
    return f"{prefix}\n{text}"


def is_noise(text: str, has_image_text: bool) -> bool:
    """12. a slide piece that is only "THANK YOU", a page number ("31") or symbols ("→ →", "≈") is never a source.

    Short pieces with real words stay ("Static budgets", "gg_season()"): with the slide title they can still answer.
    """
    if has_image_text:
        return False
    words = re.sub(r"[^A-Za-z ]+", " ", text).split()
    return len("".join(words)) < 4 or " ".join(words).lower() in {"thank you", "thanks", "questions"}


def syllabus_extras() -> dict[str, dict[str, str]]:
    """course_code -> the syllabus title and the bibliography, which the course table does not keep.

    13. the title is the name the syllabus itself uses ("ORGANIZATIONAL BEHAVIOR", "STRATEGIES FOR COMPETING IN
        INDUSTRIES AND MARKETS"), so a question that uses that name also finds the summary
    14. the bibliography answers "what book do we use?"; it is cut to its first ~300 characters
    """
    path = PATHS["syllabus_sections"]
    if not path.is_file():
        return {}
    extras: dict[str, dict[str, str]] = {}
    for section in read_jsonl(path):
        code, text = section["course_code"], section.get("text", "")
        item = extras.setdefault(code, {"title": "", "title_id": "", "books": "", "books_id": ""})
        if section["section_id"].endswith("_sec01"):
            title_lines = []
            for line in text.strip().splitlines():
                if line.lower().startswith("dual degree") or len(title_lines) == 3:
                    break
                title_lines.append(line.strip())
            item["title"], item["title_id"] = " ".join(title_lines).title(), section["section_id"]
        elif section.get("section_name") == "BIBLIOGRAPHY" and not item["books"]:
            books = re.sub(r"\s+", " ", re.sub(r"(?i)^\s*bibliography\s*|\bnull\b", " ", text)).strip()
            if len(books) > 300:
                books = books[:300].rsplit(" ", 1)[0] + " ..."
            item["books"], item["books_id"] = books, section["section_id"]
    return extras


def _course_summary(row: dict[str, str], extra: dict[str, str] | None = None) -> str:
    """One course-table row -> a short paragraph that answers the usual admin questions in plain words."""
    name, code = row["course_name"], row["course_code"]
    extra = extra or {}
    title = extra.get("title", "")
    also = f"; syllabus title: {title}" if title and title.lower() != name.lower() else ""
    parts = [f"{name} ({code}{also}): course summary from the syllabus."]
    if row.get("final_exam_weight"):
        parts.append(f"The final exam is worth {float(row['final_exam_weight']):g}% of the final grade.")
    else:
        parts.append("There is no final exam in this course.")
    components = json.loads(row.get("components") or "[]")
    if components:
        listed = ", ".join(f"{c['name']} {float(c['weight']):g}%" for c in components)
        parts.append(f"How the course is evaluated: {listed}.")
    if row.get("attendance_limit_pct"):
        parts.append(f"Minimum attendance: {float(row['attendance_limit_pct']):g}% of the sessions.")
    if row.get("retake_policy"):
        parts.append(f"Re-sit policy: {row['retake_policy']}")
    if row.get("professor"):
        professor = row["professor"].title()
        parts.append(f"Professor: {professor}, who teaches this course.")  # also answers "who teaches...?"
    if row.get("n_sessions"):
        parts.append(f"Number of sessions: {row['n_sessions']}.")
    if extra.get("books"):
        parts.append(f"Bibliography (the book list for this course): {extra['books']}")
    return " ".join(parts)


def course_summary_chunks(kb_sig: str) -> tuple[list[Chunk], list[Parent]]:
    """One chunk (and its parent) per course from data/processed/course_table.csv; nothing if the table is missing.

    8. covers = the syllabus sections the row was read from, so a question whose gold_ids are those sections
       counts this chunk as a correct source in the evaluation.
    9. the chunk_id is the same one the cross-course answers use ("course_row__<code>"), so the app shows it the same way.
    """
    path = PATHS["course_table"]
    if not path.is_file():
        return [], []
    chunks, parents = [], []
    extras = syllabus_extras()
    for row in read_csv(path):
        code = row["course_code"]
        extra = extras.get(code, {})
        covers = [s for s in row.get("evidence_section_ids", "").split(";") if s]
        covers += [extra[k] for k in ("title_id", "books_id") if extra.get(k) and extra[k] not in covers]
        outline_path = f"Course summary › {row['course_name']}"
        text = _course_summary(row, extra)
        parent_id = f"parent__course_row__{code}"
        parents.append(Parent(parent_id=parent_id, outline_path=outline_path, covers=covers, text=text))
        chunks.append(
            Chunk(
                chunk_id=f"course_row__{code}",
                kb_sig=kb_sig,
                strategy="section",
                doc_type="course_row",
                course_code=code,
                sessions=[],
                source_id=f"{code}_syl",
                covers=covers,
                parent_id=parent_id,
                outline_path=outline_path,
                text=text,
                embed_text=f"{outline_path}\n{text}",
            )
        )
    return chunks, parents
