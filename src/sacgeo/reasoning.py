"""Citation Rigor + Reasoning Blob Completeness gates (autoresearch protocol).

Adapted from dlmastery/autoresearchimage/core/reasoning.py. The validator
refuses to write a research_journal entry unless every required field meets
its word-count floor and citations are properly formatted.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


WORD_FLOORS = {
    "diagnosis": 60,
    "citation": 40,           # single-paper floor; multi-paper floor is 80
    "hypothesis": 50,
    "prediction": 25,
    "verdict": 30,
    "learning": 40,
}

# A valid citation must contain (in any reasonable order):
#   - a 4-digit year (1900–2099)
#   - a single-quoted 'Title'
#   - an arXiv or bioRxiv ID like arXiv:1234.56789
# and end with a relevance note separated by an em-dash or "--".
CITATION_RE_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
CITATION_RE_TITLE = re.compile(r"'[^']{4,}'")
CITATION_RE_ID = re.compile(r"(?:arXiv|bioRxiv):\s*\d{4}\.\d{4,5}")
CITATION_RE_DASH = re.compile(r"(?:—|--)\s*\S")


@dataclass
class ReasoningEntry:
    experiment_id: str
    title: str
    diagnosis: str
    citations: list[str]
    hypothesis: str
    prediction: str
    verdict: str = ""
    learning: str = ""
    composite: float = 0.0
    composite_fingerprint: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _wc(s: str) -> int:
    return len([w for w in s.strip().split() if w])


def validate_entry(e: ReasoningEntry, require_post: bool = False) -> list[str]:
    """Return list of validation errors. Empty list = pass."""
    errs: list[str] = []
    for fld in ("diagnosis", "hypothesis", "prediction"):
        n = _wc(getattr(e, fld))
        if n < WORD_FLOORS[fld]:
            errs.append(f"{fld}: {n} words < floor {WORD_FLOORS[fld]}")
    # citation rigor
    if not e.citations:
        errs.append("citations: empty list")
    else:
        cite_words = sum(_wc(c) for c in e.citations)
        floor = WORD_FLOORS["citation"] if len(e.citations) == 1 else 80
        if cite_words < floor:
            errs.append(f"citations: {cite_words} words < floor {floor}")
        for c in e.citations:
            missing = []
            if not CITATION_RE_YEAR.search(c):
                missing.append("year")
            if not CITATION_RE_TITLE.search(c):
                missing.append("'Title'")
            if not CITATION_RE_ID.search(c):
                missing.append("arXiv:NNNN.NNNNN")
            if not CITATION_RE_DASH.search(c):
                missing.append("--- relevance note")
            if missing:
                errs.append(
                    f"citation missing {missing}: '{c[:80]}...'"
                )
    if require_post:
        for fld in ("verdict", "learning"):
            n = _wc(getattr(e, fld))
            if n < WORD_FLOORS[fld]:
                errs.append(f"{fld}: {n} words < floor {WORD_FLOORS[fld]}")
    return errs


def append_entry(path: str | Path, entry: ReasoningEntry,
                 require_post: bool = False) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    errs = validate_entry(entry, require_post=require_post)
    if errs:
        raise ValueError(
            "Reasoning entry rejected by autoresearch gates:\n  - " +
            "\n  - ".join(errs)
        )
    rows = []
    if p.exists():
        rows = json.loads(p.read_text() or "[]")
    rows = [r for r in rows if r.get("experiment_id") != entry.experiment_id]
    rows.append(entry.to_dict())
    p.write_text(json.dumps(rows, indent=2))


def load_all(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text() or "[]")
