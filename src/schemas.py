"""Shared data formats (owner: Francesca).

Every record that passes from one person to another is defined here, once.
Writers create these objects and readers load them, so the field names always match.

Rules:
- Never rename or remove a field silently. Open a pull request titled "CONTRACT: ..."
  and get it approved by everyone who reads that record.
- Adding a field is fine, but tell the readers first.
- tests/test_contracts.py checks every produced file against these classes.

Who writes -> who reads:
    ManifestRow      data/manifest.csv                      Santiago -> all scripts
    SyllabusSection  data/interim/syllabus_sections.jsonl   Santiago -> chunking, course table, Elena
    SlideRecord      data/interim/slides.jsonl              Santiago -> chunking, Irene, Elena
    ImageText        data/image_texts.jsonl                 Santiago (extraction, OCR) + Irene (class, description) -> Santiago
    Chunk            data/processed/chunks__<kb_sig>.jsonl  Santiago -> Sergio, Elena, Irene
    Parent           data/processed/parents__<kb_sig>.jsonl Santiago -> Sergio
    CourseRow        data/processed/course_table.csv        Santiago -> Sergio
    RouterQuestion   data/labels/router_questions.csv       Francesca -> train_router
    Question         eval/questions.csv                     everyone -> Elena
    AnswerResult     returned by answer()                   Sergio -> Irene (app), Elena (evaluation)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional


class Record:
    """Base class: dict conversion with a strict field check."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, data: dict[str, Any], strict: bool = True):
        names = set(cls.field_names())
        if strict:
            extra = set(data) - names
            if extra:
                raise ValueError(f"{cls.__name__}: unknown fields {sorted(extra)}")
        return cls(**{k: v for k, v in data.items() if k in names})  # type: ignore[call-arg]


# ---------------------------------------------------------------- knowledge base

@dataclass
class ManifestRow(Record):
    source_id: str          # e.g. "c01_syl" or "c01_s04" (see src/utils/ids.py)
    course_code: str        # e.g. "c01"
    course_name: str
    doc_type: str           # "syllabus" | "slides"
    sessions: str           # "4", "5;6" or "" for a syllabus
    file: str               # path under data/raw/
    language: str           # "en" | "es"


@dataclass
class SyllabusSection(Record):
    section_id: str         # "<source_id>_sec<NN>"
    source_id: str
    course_code: str
    section_name: str       # EVALUATION, ATTENDANCE, RESIT, PROGRAM, BIBLIOGRAPHY, PROFESSOR, METHODOLOGY, AI_POLICY, OTHER
    page_start: int
    page_end: int
    text: str
    tables: list[str] = field(default_factory=list)   # each table as markdown


@dataclass
class SlideRecord(Record):
    slide_id: str           # "<source_id>_p<NNN>"
    source_id: str
    course_code: str
    sessions: list[int]
    page: int               # PDF page, from 1
    title: str
    outline: list[str]      # e.g. ["2. Topic", "2.1 Subtopic"]
    text: str
    is_agenda: bool
    is_low_text: bool
    is_garbled: bool
    n_images: int


@dataclass
class ImageText(Record):
    image_id: str           # "<slide_id>_img<NN>" or "<slide_id>_full"
    slide_id: str
    source_id: str
    kind: str               # "embedded" | "full_slide"
    image_path: str
    ocr_text: str           # Santiago
    image_class: str = ""   # Irene: diagram | chart | formula | table | photo | decoration
    class_prob: float = 0.0
    description: str = ""   # Irene: local vision model, "" if not described
    vision_model: str = ""
    human_checked: bool = False


@dataclass
class Chunk(Record):
    chunk_id: str           # "<strategy>__<first covered id>__<NN>"
    kb_sig: str
    strategy: str           # "section" | "slide"
    doc_type: str           # "syllabus_section" | "slide" | "course_row"
    course_code: str
    sessions: list[int]
    source_id: str
    covers: list[str]       # slide_ids / section_ids this chunk contains (used by the evaluation)
    parent_id: str
    outline_path: str
    text: str
    embed_text: str
    has_image_text: bool = False
    duplicate_of: Optional[str] = None


@dataclass
class Parent(Record):
    parent_id: str
    outline_path: str
    covers: list[str]
    text: str


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
    evidence_section_ids: str   # ";"-separated
    human_checked: bool = False


# ---------------------------------------------------------------- router and evaluation

@dataclass
class RouterQuestion(Record):
    question_id: str
    question: str
    label: str              # course_admin | cross_course | course_content | out_of_scope
    origin: str             # written | paraphrase
    author: str


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

@dataclass
class Filters(Record):
    course_code: Optional[str] = None
    sessions: list[int] = field(default_factory=list)


@dataclass
class RouteDecision(Record):
    label: str              # course_admin | cross_course | course_content | out_of_scope
    prob: float
    source: str             # "classifier" | "llm" | "none"


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

    @classmethod
    def from_dict(cls, data: dict[str, Any], strict: bool = True):
        data = dict(data)
        if isinstance(data.get("chunk"), dict):
            data["chunk"] = Chunk.from_dict(data["chunk"], strict=strict)
        return super().from_dict(data, strict=strict)


@dataclass
class Citation(Record):
    label: str              # "S1"
    chunk_id: str
    covers: list[str]


@dataclass
class AnswerResult(Record):
    """What answer() returns. The app and the evaluation read only these fields."""
    question: str
    rewritten_question: str
    route: RouteDecision
    filters_used: Filters
    context_hits: list[Hit]             # the numbered sources S1, S2, ...
    answer_text: str
    citations: list[Citation]
    refused: bool
    timings_ms: dict[str, float]        # rewrite, route, retrieve, rerank, generate, total
    config_name: str
    kb_sig: str

    @classmethod
    def from_dict(cls, data: dict[str, Any], strict: bool = True):
        data = dict(data)
        data["route"] = RouteDecision.from_dict(data["route"], strict=strict)
        data["filters_used"] = Filters.from_dict(data["filters_used"], strict=strict)
        data["context_hits"] = [Hit.from_dict(h, strict=strict) for h in data["context_hits"]]
        data["citations"] = [Citation.from_dict(c, strict=strict) for c in data["citations"]]
        return super().from_dict(data, strict=strict)


REFUSAL = "This isn't covered in the course material I have."

# file -> record class, used by tests/test_contracts.py
CONTRACT_FILES = {
    "data/manifest.csv": ManifestRow,
    "data/interim/syllabus_sections.jsonl": SyllabusSection,
    "data/interim/slides.jsonl": SlideRecord,
    "data/image_texts.jsonl": ImageText,
    "data/processed/course_table.csv": CourseRow,
    "data/labels/router_questions.csv": RouterQuestion,
    "eval/questions.csv": Question,
}
