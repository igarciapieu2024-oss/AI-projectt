# Shared data formats: every record that one step writes and another step reads is
# defined here, once. This is done so the
# field names always match across the project.
#
# Record           File                                    Written by -> read by
#   ManifestRow      data/manifest.csv                       written by hand -> all parsing steps
#   SyllabusSection  data/interim/syllabus_sections.jsonl    parse_syllabi.py -> course table, chunking
#   SlideRecord      data/interim/slides.jsonl               parse_slides.py -> image step, chunking
#   ImageText        data/image_texts.jsonl                  src/vision/ (OCR, class, description) -> chunking
#   Chunk            data/processed/chunks__<kb_sig>.jsonl   chunking.py -> indexes, search, app, evaluation
#   Parent           data/processed/parents__<kb_sig>.jsonl  chunking.py
#   CourseRow        data/processed/course_table.csv         course_table.py -> table_qa.py, chunking
#   RouterQuestion   data/labels/router_questions.csv        written by hand -> train_router.py
#   Question         eval/questions*.csv                     written by hand -> evaluation
#   AnswerResult     returned by answer()                    answer.py -> app, evaluation
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional


# Base class of every record: converts it to and from a dictionary, the form used
# to save it as one line of JSON or one row of CSV.
class Record:

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]  # type: ignore[arg-type]

    # Build a record from a dictionary. With strict=True, a field that the class
    # does not define raises an error, so a renamed or misspelled field in a file
    # is detected at once instead of being silently ignored.
    @classmethod
    def from_dict(cls, data: dict[str, Any], strict: bool = True):
        names = set(cls.field_names())
        if strict:
            extra = set(data) - names
            if extra:
                raise ValueError(f"{cls.__name__}: unknown fields {sorted(extra)}")
        return cls(**{k: v for k, v in data.items() if k in names})  # type: ignore[call-arg]


# ---------------------------------------------------------------- knowledge base

# One row of data/manifest.csv: one PDF and what it is.
@dataclass
class ManifestRow(Record):
    source_id: str          # example: "c01_syl" or "c01_s04" (see src/utils/ids.py)
    course_code: str        # example: "c01"
    course_name: str
    doc_type: str           # "syllabus" | "slides"
    sessions: str           # example:"[4]", "[5, 6]", or "" for a syllabus
    file: str               # path under data/raw/
    language: str           # "en" | "es"


# One section of a syllabus.
@dataclass
class SyllabusSection(Record):
    section_id: str         # "<source_id>_sec<NN>"
    source_id: str
    course_code: str
    section_name: str       # EVALUATION, ATTENDANCE, RESIT, PROGRAM, BIBLIOGRAPHY, PROFESSOR, METHODOLOGY, AI_POLICY, OTHER
    page_start: int
    page_end: int
    text: str
    tables: list[str] = field(default_factory=list)   # each table as Markdown


# One slide, one page of a slide deck.
@dataclass
class SlideRecord(Record):
    slide_id: str           # "<source_id>_p<NNN>"
    source_id: str
    course_code: str
    sessions: list[int]
    page: int               # PDF page, from 1
    title: str
    outline: list[str]      # example: ["2. Topic", "2.1 Subtopic"]
    text: str
    is_agenda: bool
    is_low_text: bool
    is_garbled: bool
    n_images: int


# The text obtained from one image of a slide: the OCR text and, for useful
# images, the class and description given by the vision model.
@dataclass
class ImageText(Record):
    image_id: str           # "<slide_id>_img<NN>" or "<slide_id>_full"
    slide_id: str
    source_id: str
    kind: str               # "embedded" (an image inside the slide) | "full_slide" (the whole page)
    image_path: str
    ocr_text: str
    image_class: str = ""   # diagram | chart | formula | table | photo | decoration
    class_prob: float = 0.0
    description: str = ""   # description by the vision model, "" if not described
    vision_model: str = ""
    human_checked: bool = False


# One searchable piece of text. text is what the language model reads;
# embed_text adds the course name and position in front and is what gets embedded
# and searched.
@dataclass
class Chunk(Record):
    chunk_id: str           # "<strategy>__<first covered id>__<NN>"
    kb_sig: str             # knowledge base the chunk belongs to
    strategy: str           # "section" | "slide"
    doc_type: str           # "syllabus_section" | "slide" | "course_row"
    course_code: str
    sessions: list[int]
    source_id: str
    covers: list[str]       # slide_ids / section_ids this chunk contains (used by the evaluation)
    parent_id: str
    outline_path: str       # position in the course, e.g. "Session 9 > slide title"
    text: str
    embed_text: str
    has_image_text: bool = False
    duplicate_of: Optional[str] = None   # id of the chunk it duplicates, set in step 6


# The whole slide or section a chunk was cut from.
@dataclass
class Parent(Record):
    parent_id: str
    outline_path: str
    covers: list[str]
    text: str


# One row of the course table: the admin facts of one course,
# taken from its syllabus.
@dataclass
class CourseRow(Record):
    course_code: str
    course_name: str
    final_exam_weight: Optional[float]
    components: str         # JSON list of {"name": ..., "weight": ...}
    attendance_limit_pct: Optional[float]
    retake_policy: str
    professor: str
    n_sessions: Optional[int]
    language: str
    evidence_section_ids: str   # ";"-separated ids of the syllabus sections the values come from
    human_checked: bool = False


# ---------------------------------------------------------------- router and evaluation

# One labelled question used to train the router.
@dataclass
class RouterQuestion(Record):
    question_id: str
    question: str
    label: str              # course_admin | cross_course | course_content | out_of_scope
    origin: str             # written | paraphrase
    author: str


# One evaluation question with its correct sources and reference answer.
@dataclass
class Question(Record):
    question_id: str
    split: str              # dev | test
    type: str               # admin | cross | content | image | followup | unanswerable
    turn1: str              # first question of a follow-up pair, else ""
    question: str
    course_code: str
    gold_ids: str           # ";"-separated slide_ids / section_ids, "" if unanswerable
    reference_answer: str
    key_facts: str          # ";"-separated
    answerable: int         # 1 | 0
    author: str


# ---------------------------------------------------------------- answer() contract

# Restricts the search to one course and, optionally, to some sessions.
@dataclass
class Filters(Record):
    course_code: Optional[str] = None
    sessions: list[int] = field(default_factory=list)


# The router's decision for a question.
@dataclass
class RouteDecision(Record):
    label: str              # course_admin | cross_course | course_content | out_of_scope
    prob: float
    source: str             # "classifier" | "llm" | "none"


# One search result: the chunk with its position and the score of each search stage.
@dataclass
class Hit(Record):
    chunk: Chunk
    rank: int
    score_dense: Optional[float] = None
    score_bm25: Optional[float] = None
    score_rrf: Optional[float] = None
    score_rerank: Optional[float] = None
    label: str = ""         # "S1", "S2", ... as cited in the answer
    thumbnail_path: str = ""

    # The chunk is stored as a nested dictionary, so it is rebuilt as a Chunk first.
    @classmethod
    def from_dict(cls, data: dict[str, Any], strict: bool = True):
        data = dict(data)
        if isinstance(data.get("chunk"), dict):
            data["chunk"] = Chunk.from_dict(data["chunk"], strict=strict)
        return super().from_dict(data, strict=strict)


# One source cited in the answer.
@dataclass
class Citation(Record):
    label: str              # "S1"
    chunk_id: str
    covers: list[str]


# What answer() returns. The app and the evaluation read only these fields.
@dataclass
class AnswerResult(Record):
    question: str
    rewritten_question: str
    route: RouteDecision
    filters_used: Filters
    context_hits: list[Hit]             # the numbered sources as S1, S2, 
    answer_text: str
    citations: list[Citation]
    refused: bool
    timings_ms: dict[str, float]        # rewrite, route, retrieve, rerank, generate, total
    config_name: str
    kb_sig: str

    # The nested records (route, filters, sources, citations) are rebuilt
    # from their dictionaries before the result itself.
    @classmethod
    def from_dict(cls, data: dict[str, Any], strict: bool = True):
        data = dict(data)
        data["route"] = RouteDecision.from_dict(data["route"], strict=strict)
        data["filters_used"] = Filters.from_dict(data["filters_used"], strict=strict)
        data["context_hits"] = [Hit.from_dict(h, strict=strict) for h in data["context_hits"]]
        data["citations"] = [Citation.from_dict(c, strict=strict) for c in data["citations"]]
        return super().from_dict(data, strict=strict)


# The fixed answer given when the material does not support an answer.
REFUSAL = "This isn't covered in the course material I have."

# file is the record class, used by tests/test_contracts.py to check every file's fields
CONTRACT_FILES = {
    "data/manifest.csv": ManifestRow,
    "data/interim/syllabus_sections.jsonl": SyllabusSection,
    "data/interim/slides.jsonl": SlideRecord,
    "data/image_texts.jsonl": ImageText,
    "data/processed/course_table.csv": CourseRow,
    "data/labels/router_questions.csv": RouterQuestion,
    "eval/questions.csv": Question,
}
