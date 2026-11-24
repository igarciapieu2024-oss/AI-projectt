# Text helpers for the knowledge base: readable syllabus tables, cutting text at sentence ends, and one summary chunk
# per course.
# PyMuPDF reads the syllabus tables with spaces and decimal points missing, which is hard to match for the search and
# hard to read for the LLM, so we rewrite each table row as one clean line. Only the text changes: section_ids and
# slide_ids stay the same, so the gold_ids of the evaluation still match. chunking.py imports all the public functions
# of this file.
from __future__ import annotations

# json reads the evaluation components of the course table; re is for all the text patterns below.
import json
import re

# wordninja splits words that the PDF glued together, using an English word list. It's optional: without it we still
# clean the tables, only the glued words stay glued.
try:
    import wordninja
except ImportError:
    wordninja = None

# The folders of the project, the formats of a chunk and its parent, and the CSV and JSONL readers.
from src.config import PATHS
from src.schemas import Chunk, Parent
from src.utils.io import read_csv, read_jsonl

# ----------------------------------------------------------------------
# Table cleaning

# We only check runs of at least 8 letters for glued words; shorter words are left alone.
GLUED_MIN_LETTERS = 8
# A manual correction for the one split the word list gets wrong in our syllabi.
SPLIT_FIXES = {"coverall": "cover all"}
# Short words that are fine inside a split. Any other part shorter than 3 letters means the split is wrong, so we keep
# the word as it is.
SHORT_WORDS = {"a", "i", "an", "as", "at", "be", "by", "do", "in", "is", "it", "of", "on", "or", "to", "we", "up", "no"}
# A table cell that only holds a weight. We use it to recognise evaluation and workload rows in table_to_text().
PERCENT_CELL = re.compile(r"^\d{1,4}\s?%(\s?\.)?$")
# The PDF loses the decimal point and leaves a loose " ." after the number:
# "133% ." is really 13.3%. These patterns put the decimal point back.
LOST_DECIMAL_PERCENT = re.compile(r"\b(\d+)(\d)\s?%\s?\.")
LOST_DECIMAL_HOURS = re.compile(r"\b(\d+)(\d)\s?hours\s?\.")
LOST_DECIMAL_LOOSE = re.compile(r"(?<=\s)\.\s(\d)(\d)(?!\d)")


# Split one glued word into its words. When the split looks doubtful we return the word unchanged, because a wrong
# split damages a real word, while a glued word left as it is only loses some matching.
def _split_glued(word: str) -> str:
    # Without the library, or with a short word, there is nothing to split.
    if wordninja is None or len(word) < GLUED_MIN_LETTERS:
        return word
    # The words the library finds inside the glued word.
    parts = wordninja.split(word)
    if len(parts) < 2 or any(len(p) < 3 and p.lower() not in SHORT_WORDS for p in parts):
        return word  # a real long word or a doubtful split
    # Join the parts with spaces, applying our manual correction where needed.
    return " ".join(SPLIT_FIXES.get(p.lower(), p) for p in parts)


# Make one table cell readable. We restore the decimals first, before the steps that move spaces around and would
# break the " ." pattern the regular expressions look for.
def clean_cell(text: str) -> str:
    # First the three patterns that put lost decimal points back.
    text = LOST_DECIMAL_PERCENT.sub(r"\1.\2%", text)
    text = LOST_DECIMAL_HOURS.sub(r"\1.\2 hours", text)
    text = LOST_DECIMAL_LOOSE.sub(r"\1.\2", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)  # separate CamelCase words
    text = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", text)  # separate letters from numbers
    # Every run of letters goes through _split_glued(), which splits it only when it is a glued word.
    text = re.sub(r"[A-Za-z]+", lambda m: _split_glued(m.group(0)), text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)  # no space before punctuation
    text = re.sub(r"(?<=\w)\s*-\s+(?=\w)", " ", text)  # stray hyphen between two words
    # Single spaces, and no spaces or bars left at the ends.
    return re.sub(r"\s+", " ", text).strip(" |")


# Return the cells of each row of a markdown table, skipping the separator lines.
def _rows(table: str) -> list[list[str]]:
    rows = []
    for line in table.splitlines():
        # A line that doesn't start with "|" isn't a table row, and a line made only of bars and dashes is a
        # separator.
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|- "):
            continue
        # The cells of the row, without the outer bars.
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


# Turn a syllabus table into one readable line per row. Evaluation rows keep the component, the weight and the
# comment; we drop the learning-objectives column because it repeats the same text in every row. Workload rows keep
# the activity, the share of time and the hours. The rows of any other table are joined into a single line.
def table_to_text(table: str) -> str:
    # The rows of the table; without rows there is no text.
    rows = _rows(table)
    if not rows:
        return ""
    lines = []
    for cells in rows:
        # Clean every cell of the row.
        cleaned = [clean_cell(c) for c in cells]
        # We check the weight on the raw cell, because PERCENT_CELL is written for the format the PDF produces.
        if len(cleaned) >= 2 and PERCENT_CELL.match(cells[1]):
            # The name of the component (or activity) and its weight.
            name, weight = cleaned[0].rstrip(" ,"), cleaned[1]
            if len(cleaned) >= 4:  # evaluation row
                comment = cleaned[3]
                lines.append(f"- {name}: {weight}." + (f" {comment}" if comment else ""))
            elif len(cleaned) == 3:  # workload row
                lines.append(f"- {name}: {weight} of the time, {cleaned[2]}")
            else:
                lines.append(f"- {name}: {weight}")
        elif any(c.lower() in {"criteria", "learning activity"} for c in cleaned):
            continue  # header row: each line above already says what its values are
        # Any other row: its cells joined with " | ".
        elif any(cleaned):
            lines.append(" | ".join(c for c in cleaned if c))
    return "\n".join(lines)


# Return the text of one syllabus section for chunking: its body plus its tables written as readable lines.
# chunking.py calls it for every syllabus section.
def section_text(section: dict) -> str:
    # The body of the section, with the words split by a hyphen at a line end joined again.
    text = fix_broken_hyphens(section.get("text", ""))
    # Each table as readable lines; empty ones are dropped.
    tables = [table_to_text(t) for t in section.get("tables", [])]
    tables = [t for t in tables if t]
    # The tables go after the body, separated by blank lines.
    if tables:
        text += "\n\n" + "\n\n".join(tables)
    return text


# ----------------------------------------------------------------------
# Chunking

# The end of a sentence: ".", "?" or "!" followed by a space and the start of a new sentence.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(\[-])")
# Where a new session starts in the syllabus programme ("SESSION 4", "SESSIONS 9-10-11").
SESSION_START = re.compile(r"(?=\bSESSIONS?\s+\d)")
SESSION_NUMBERS = re.compile(r"\bSESSIONS?\s+((?:\d+\s*(?:-|,|&|and)?\s*)+)")


# Cut a single sentence that is longer than a chunk at its spaces, so no word is broken.
def _fit_words(text: str, size: int) -> list[str]:
    # words: the words of the sentence; pieces: the finished pieces; current: the piece being filled.
    words, pieces, current = text.split(), [], ""
    for word in words:
        # If the word doesn't fit, we close the current piece and start a new one with this word.
        if current and len(current) + 1 + len(word) > size:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    # The last piece is added too.
    return pieces + ([current] if current else [])


# Cut a text into chunks of at most `size` characters at sentence ends, never inside a word. A fixed cut every `size`
# characters gave chunks that started in the middle of a word, so we pack whole sentences (and whole table lines)
# together until the next one doesn't fit.
# `overlap`: the last sentences of a chunk, up to that many characters, are repeated at the start of the next one, so
# an idea split between two chunks is still complete in one of them.
# `by_session` is used for the syllabus programme: the units are whole sessions instead of sentences, with no overlap,
# so each chunk holds complete sessions and can be tagged with their numbers by session_numbers(). chunking.py calls
# it for both syllabus sections and slides.
def split_into_chunks(text: str, size: int, overlap: int, by_session: bool = False) -> list[str]:
    # The units that will be packed into chunks.
    units = []
    if by_session:
        # For the programme, a block is everything from one "SESSION N" to the next, and there is no overlap.
        blocks = [b for b in SESSION_START.split(text or "") if b.strip()]
        overlap = 0
    else:
        # The PDF breaks sentences at the end of each printed line, so we join the lines back into paragraphs. An
        # empty line, or a line starting with "- " (a bullet or a table line), starts a new block.
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

    # Turn the blocks into units: a whole session, or a table or bullet line that fits, stays as one unit; any other
    # block is split into sentences. Units longer than a chunk are cut at spaces.
    for block in blocks:
        # Collapse the spaces of the block.
        block = " ".join(block.split())
        # A whole session, or a bullet or table line that fits in a chunk, stays in one piece.
        whole = by_session or (block.startswith("- ") and len(block) <= size)
        sentences = [block] if whole else SENTENCE_END.split(block)
        for sentence in sentences:
            units.extend([sentence] if len(sentence) <= size else _fit_words(sentence, size))

    # Pack the units into chunks.
    chunks, current = [], []
    for unit in units:
        # If adding this unit would go over the size, we close the current chunk.
        if current and len(" ".join(current + [unit])) > size:
            chunks.append(" ".join(current))
            # We carry over the last sentences of the chunk we just closed, up to `overlap` characters.
            carried = []
            for previous in reversed(current):
                if len(" ".join([previous] + carried)) > overlap:
                    break
                carried.insert(0, previous)
            # We drop the overlap if it would leave no room for the next unit.
            current = carried if len(" ".join(carried + [unit])) <= size else []
        # The unit goes into the current chunk.
        current.append(unit)
    # The last chunk is added too.
    if current:
        chunks.append(" ".join(current))
    return chunks


# Return the session numbers a programme chunk covers. A chunk that doesn't start with "SESSION" continues the last
# session of the previous chunk, so it keeps that number. chunking.py stores the result in the chunk's `sessions`
# field.
def session_numbers(piece: str, previous: list[int]) -> list[int]:
    numbers = []
    # Every number written after "SESSION" or "SESSIONS".
    for match in SESSION_NUMBERS.finditer(piece):
        numbers += [int(n) for n in re.findall(r"\d+", match.group(1))]
    # A piece that doesn't start with "SESSION" continues the last session of the previous piece.
    if previous and not re.match(r"\s*SESSIONS?\s+\d", piece):
        numbers = [previous[-1]] + numbers
    # Values outside 1-99 aren't session numbers.
    return sorted(set(n for n in numbers if 0 < n < 100))


# Join again the words the PDF split with a hyphen at a line end. "and", "or" and "to" are left out, so expressions
# such as "short- and long-term" keep their space.
def fix_broken_hyphens(text: str) -> str:
    return re.sub(r"(?<=[a-z])-[ \t]*\n?[ \t]*(?=[a-z]{2,}\b)(?!and\b|or\b|to\b)", "-", text)


# ----------------------------------------------------------------------
# Course names and headings for the searched text

# Map course_code -> course name from data/manifest.csv. chunking.py passes the name
# to with_course().
def course_names() -> dict[str, str]:
    # Without the manifest there are no names.
    path = PATHS["manifest"]
    if not path.is_file():
        return {}
    return {row["course_code"]: row["course_name"] for row in read_csv(path)}


# Return the real title of a syllabus section, which chunking.py uses in the outline path. 36 of the 92 sections are
# labelled OTHER even though they have a clear heading, and an outline such as "Syllabus › OTHER" tells the search
# nothing. We use the first line when it looks like a heading: short, no colon and mostly capitals. The first section
# of each syllabus is the course header, so we call it "Course details".
def section_heading(section: dict) -> str:
    # The first line of the section, without a final colon.
    first = section.get("text", "").strip().split("\n", 1)[0].strip().rstrip(":")
    # Its letters, to measure how many are capitals.
    letters = [ch for ch in first if ch.isalpha()]
    looks_like_heading = 0 < len(first) <= 60 and ":" not in first and letters and (
        sum(ch.isupper() for ch in letters) / len(letters) > 0.6
    )
    # The first section of every syllabus is the course header.
    if section.get("section_id", "").endswith("_sec01"):
        return "Course details"
    # .title() puts acronyms and hyphenated words in lower case, so we fix the common cases back.
    if looks_like_heading:
        return first.title().replace("Ai ", "AI ").replace("Re-Sit", "Re-sit").replace("Re-Take", "Re-take")
    # Otherwise, the section name given by parse_syllabi.py, written in title case.
    return section.get("section_name", "Section").replace("_", " ").title().replace("Ai ", "AI ")


# Build the text that we embed, search and rerank: course name + outline + text. Many chunks never said which course
# they belong to, so without the course filter the search couldn't tell similar chunks from different courses apart.
# Adding the course name lets a question that names the course match the right chunks. chunking.py uses it as the
# chunk's embed_text.
def with_course(course_name: str, outline_path: str, text: str) -> str:
    # The course name goes first when we have it.
    prefix = f"{course_name} › {outline_path}" if course_name else outline_path
    return f"{prefix}\n{text}"


# True for a slide piece that can never be a useful source: a closing line such as "Thank you", a page number or only
# symbols. We keep short pieces with real words, because together with the slide title they can still answer a
# question, and we always keep a piece with text from its images. chunking.py skips the pieces marked here.
def is_noise(text: str, has_image_text: bool) -> bool:
    # A piece with image text is never noise.
    if has_image_text:
        return False
    # Keep only the letters and split them into words.
    words = re.sub(r"[^A-Za-z ]+", " ", text).split()
    # Noise: fewer than 4 letters in total, or just a closing line.
    return len("".join(words)) < 4 or " ".join(words).lower() in {"thank you", "thanks", "questions"}


# ----------------------------------------------------------------------
# Course summary chunks

# Map course_code to the syllabus title and the bibliography, which the course table doesn't keep. The title is the
# name the syllabus itself uses, so a question that uses that name also finds the summary. The bibliography answers
# questions about the course books, and we cut it to about 300 characters to keep the summary short. We keep the
# section ids so the summary can list them in covers.
def syllabus_extras() -> dict[str, dict[str, str]]:
    # Without the sections file there is nothing to read.
    path = PATHS["syllabus_sections"]
    if not path.is_file():
        return {}
    extras: dict[str, dict[str, str]] = {}
    for section in read_jsonl(path):
        # The course and the text of this section.
        code, text = section["course_code"], section.get("text", "")
        # One entry per course, created empty the first time we see the course.
        item = extras.setdefault(code, {"title": "", "title_id": "", "books": "", "books_id": ""})
        if section["section_id"].endswith("_sec01"):
            # The title is the first lines of the course header, up to three lines or until
            # the "Dual Degree" programme line.
            title_lines = []
            for line in text.strip().splitlines():
                if line.lower().startswith("dual degree") or len(title_lines) == 3:
                    break
                title_lines.append(line.strip())
            item["title"], item["title_id"] = " ".join(title_lines).title(), section["section_id"]
        elif section.get("section_name") == "BIBLIOGRAPHY" and not item["books"]:
            # Remove the "Bibliography" heading and the "null" values left by the parser.
            books = re.sub(r"\s+", " ", re.sub(r"(?i)^\s*bibliography\s*|\bnull\b", " ", text)).strip()
            # Cut it at about 300 characters, at the end of a word.
            if len(books) > 300:
                books = books[:300].rsplit(" ", 1)[0] + " ..."
            item["books"], item["books_id"] = books, section["section_id"]
    return extras


# Turn one row of the course table into a short paragraph that answers the usual admin questions (final exam,
# evaluation, attendance, re-sit, professor, sessions, books) in plain words.
def _course_summary(row: dict[str, str], extra: dict[str, str] | None = None) -> str:
    # The name and the code of the course.
    name, code = row["course_name"], row["course_code"]
    extra = extra or {}
    title = extra.get("title", "")
    # We only add the syllabus title when it's different from the course name.
    also = f"; syllabus title: {title}" if title and title.lower() != name.lower() else ""
    parts = [f"{name} ({code}{also}): course summary from the syllabus."]
    # ":g" drops the trailing ".0", so the weights read naturally.
    if row.get("final_exam_weight"):
        parts.append(f"The final exam is worth {float(row['final_exam_weight']):g}% of the final grade.")
    else:
        parts.append("There is no final exam in this course.")
    # The course table stores the evaluation components as a JSON list.
    components = json.loads(row.get("components") or "[]")
    if components:
        listed = ", ".join(f"{c['name']} {float(c['weight']):g}%" for c in components)
        parts.append(f"How the course is evaluated: {listed}.")
    # The other facts are only written when the table has them.
    if row.get("attendance_limit_pct"):
        parts.append(f"Minimum attendance: {float(row['attendance_limit_pct']):g}% of the sessions.")
    if row.get("retake_policy"):
        parts.append(f"Re-sit policy: {row['retake_policy']}")
    if row.get("professor"):
        professor = row["professor"].title()
        parts.append(f"Professor: {professor}, who teaches this course.")  # wording also matches "who teaches" questions
    if row.get("n_sessions"):
        parts.append(f"Number of sessions: {row['n_sessions']}.")
    if extra.get("books"):
        parts.append(f"Bibliography (the book list for this course): {extra['books']}")
    return " ".join(parts)


# Build one chunk and its parent per course from data/processed/course_table.csv. chunking.py calls it when the config
# switches on course_rows, and it returns nothing if the table is missing. covers lists the syllabus sections the row
# was read from, so a question whose gold_ids are those sections counts this chunk as a correct source in the
# evaluation. The chunk_id "course_row__<code>" is the same one table_qa.py uses for cross-course answers, so the app
# shows both in the same way.
def course_summary_chunks(kb_sig: str) -> tuple[list[Chunk], list[Parent]]:
    # Without the course table there are no summaries.
    path = PATHS["course_table"]
    if not path.is_file():
        return [], []
    chunks, parents = [], []
    extras = syllabus_extras()
    # One chunk and one parent per row of the table.
    for row in read_csv(path):
        code = row["course_code"]
        extra = extras.get(code, {})
        # The syllabus sections the row was read from, plus the title and bibliography sections.
        covers = [s for s in row.get("evidence_section_ids", "").split(";") if s]
        covers += [extra[k] for k in ("title_id", "books_id") if extra.get(k) and extra[k] not in covers]
        # The position, the text and the parent id of this summary.
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
                # The course name is already in outline_path, so we don't need with_course() here.
                embed_text=f"{outline_path}\n{text}",
            )
        )
    return chunks, parents
