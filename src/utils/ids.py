# Builders for every id used in the project, one function per id type.
# Each id is built only from fixed data (course, sessions, page or section number),
# never from the reading order or the date, so it is the same on every rebuild.
# All files link their records through these ids, and the evaluation questions use
# them to name their correct source.
#
#   source_id   c01_syl  |  c01_s04  |  c01_s05-06
#   section_id  c01_syl_sec04
#   slide_id    c01_s04_p025
#   image_id    c01_s04_p025_img01   |  c01_s04_p025_full
#   chunk_id    slide__c01_s04_p025__00
#   run_id      final__test__20261115-1030
#
# Numbers are zero-padded (04, 025) so every id of the same type has the same length
# and ids sort in page and session order.
from __future__ import annotations

from datetime import datetime


# Id of a document: the course code plus "_syl" for a syllabus, or "_s" plus its
# sessions for a slide deck (several sessions are joined with "-").
def source_id(course_code: str, doc_type: str, sessions: list[int] | None = None) -> str:
    if doc_type == "syllabus":
        return f"{course_code}_syl"
    if not sessions:  # a slide id without sessions would not be unique
        raise ValueError("slides need at least one session number")
    return f"{course_code}_s" + "-".join(f"{s:02d}" for s in sessions)


# Id of a syllabus section: the document id plus the section number.
def section_id(source_id_: str, n: int) -> str:
    return f"{source_id_}_sec{n:02d}"


# Id of a slide: the document id plus the page number.
def slide_id(source_id_: str, page: int) -> str:
    return f"{source_id_}_p{page:03d}"


# Id of an image: the slide id plus "_imgNN" for an image inside the slide,
# or "_full" when the whole slide is used as one image.
def image_id(slide_id_: str, n: int | None = None) -> str:
    return f"{slide_id_}_full" if n is None else f"{slide_id_}_img{n:02d}"


# Id of a chunk: the chunking strategy, the id of the first slide or section it
# covers, and the number of the piece (a long slide or section gives several chunks).
def chunk_id(strategy: str, first_covered_id: str, n: int) -> str:
    return f"{strategy}__{first_covered_id}__{n:02d}"


# Id of an evaluation run: config name, question split and the date and time
# (the current time if none is given).
def run_id(config_name: str, split: str, when: datetime | None = None) -> str:
    when = when or datetime.now()
    return f"{config_name}__{split}__{when:%Y%m%d-%H%M}"
