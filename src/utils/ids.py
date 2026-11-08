"""One builder per ID pattern. Every file joins on these IDs, so they never change.

    source_id   c01_syl  |  c01_s04  |  c01_s05-06
    section_id  c01_syl_sec04
    slide_id    c01_s04_p025
    image_id    c01_s04_p025_img01   |  c01_s04_p025_full
    chunk_id    slide__c01_s04_p025__00
    run_id      final__test__20261115-1030
"""
from __future__ import annotations

from datetime import datetime


def source_id(course_code: str, doc_type: str, sessions: list[int] | None = None) -> str:
    if doc_type == "syllabus":
        return f"{course_code}_syl"
    if not sessions:
        raise ValueError("slides need at least one session number")
    return f"{course_code}_s" + "-".join(f"{s:02d}" for s in sessions)


def section_id(source_id_: str, n: int) -> str:
    return f"{source_id_}_sec{n:02d}"


def slide_id(source_id_: str, page: int) -> str:
    return f"{source_id_}_p{page:03d}"


def image_id(slide_id_: str, n: int | None = None) -> str:
    return f"{slide_id_}_full" if n is None else f"{slide_id_}_img{n:02d}"


def chunk_id(strategy: str, first_covered_id: str, n: int) -> str:
    return f"{strategy}__{first_covered_id}__{n:02d}"


def run_id(config_name: str, split: str, when: datetime | None = None) -> str:
    when = when or datetime.now()
    return f"{config_name}__{split}__{when:%Y%m%d-%H%M}"
