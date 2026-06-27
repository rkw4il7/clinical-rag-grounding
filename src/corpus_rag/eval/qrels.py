"""Gold relevance set (qrels) loading + document matching (Layer 1).

A qrels file is JSON (consistent with ``CORPUS_SOURCES``; avoids a YAML
dependency). Schema::

    [
      {
        "query": "warfarin INR target range",
        "relevant": [
          {"contains": "target INR of 2.0 to 3.0"},
          {"source": "anticoag.pdf", "heading": "Monitoring"}
        ]
      }
    ]

Each ``relevant`` entry is a :class:`RelevanceSpec`. A retrieved document matches
a spec when:
- ``contains``: the (case-insensitive) substring appears in ``doc.content`` — the
  robust default, tolerant of HybridChunker boundary drift; and/or
- ``source`` / ``heading``: the value appears in the document's provenance
  metadata. ``source`` is matched against any ``meta`` value containing it;
  ``heading`` against the joined heading path.

Matching by content/provenance (not row id) keeps qrels stable across re-ingest.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from haystack import Document


@dataclass(frozen=True)
class RelevanceSpec:
    """One gold-relevant item, identified by content and/or provenance."""

    contains: str | None = None
    source: str | None = None
    heading: str | None = None

    def __post_init__(self) -> None:
        if not (self.contains or self.source or self.heading):
            raise ValueError("RelevanceSpec needs at least one of: contains, source, heading")


@dataclass(frozen=True)
class EvalCase:
    """A query plus its gold-relevant specs."""

    query: str
    relevant: tuple[RelevanceSpec, ...]

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("EvalCase.query must be non-empty")
        if not self.relevant:
            raise ValueError(f"EvalCase {self.query!r} has no relevant specs")


def _meta_values(meta: dict[str, Any]) -> list[str]:
    """Flatten a Haystack ``Document.meta`` dict to a list of string values."""
    out: list[str] = []

    def _walk(value: Any) -> None:
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                _walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                _walk(v)
        elif value is not None:
            out.append(str(value))

    _walk(meta or {})
    return out


def doc_matches(doc: Document, spec: RelevanceSpec) -> bool:
    """Return True if ``doc`` satisfies every populated field of ``spec``.

    Fields are ANDed: a spec with both ``contains`` and ``source`` requires both.
    """
    if spec.contains is not None:
        if spec.contains.lower() not in (doc.content or "").lower():
            return False
    if spec.source is not None or spec.heading is not None:
        meta_blob = " \n ".join(_meta_values(doc.meta)).lower()
        if spec.source is not None and spec.source.lower() not in meta_blob:
            return False
        if spec.heading is not None and spec.heading.lower() not in meta_blob:
            return False
    return True


def relevance_flags(docs: Sequence[Document], specs: Sequence[RelevanceSpec]) -> list[bool]:
    """Per-rank relevance: ``flags[i]`` is True iff ``docs[i]`` matches any spec."""
    return [any(doc_matches(d, s) for s in specs) for d in docs]


def relevance_gains(docs: Sequence[Document], specs: Sequence[RelevanceSpec]) -> list[bool]:
    """Per-rank binary gains crediting each gold spec at most once.

    Each spec is credited at the FIRST retrieved document that covers it; later
    docs covering an already-credited spec score 0. This keeps the total earned
    gain ≤ the number of distinct specs, so nDCG (whose ideal DCG is computed over
    that spec count) stays in [0, 1] even when several docs match one spec.
    Distinct from :func:`relevance_flags` (per-document, any-spec match).
    """
    covered: set[int] = set()
    gains: list[bool] = []
    for d in docs:
        gain = False
        for i, s in enumerate(specs):
            if i not in covered and doc_matches(d, s):
                covered.add(i)
                gain = True
                break
        gains.append(gain)
    return gains


def specs_covered(docs: Sequence[Document], specs: Sequence[RelevanceSpec]) -> int:
    """Count distinct gold specs with >=1 matching document in ``docs``.

    Used for recall (fraction of gold items the retrieval surfaced), which is
    distinct from the count of relevant documents retrieved.
    """
    return sum(1 for s in specs if any(doc_matches(d, s) for d in docs))


def parse_qrels(raw: list[dict[str, Any]]) -> list[EvalCase]:
    """Parse a decoded qrels list into :class:`EvalCase` objects."""
    cases: list[EvalCase] = []
    for i, entry in enumerate(raw):
        try:
            query = entry["query"]
            specs = tuple(RelevanceSpec(**r) for r in entry["relevant"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"qrels entry {i} is malformed: {exc}") from exc
        cases.append(EvalCase(query=query, relevant=specs))
    return cases


def load_qrels(path: str | Path) -> list[EvalCase]:
    """Load + validate a JSON qrels file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("qrels file must contain a JSON array of cases")
    return parse_qrels(data)
