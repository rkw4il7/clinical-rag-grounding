"""Tests for the Streamlit app's pure helpers.

Only import-safe, side-effect-free logic is unit-tested here; the Streamlit
rendering path is exercised manually / via the live acceptance run.
"""

from __future__ import annotations

from corpus_rag.app import _FIRST_LINE_MAX, first_line


def test_first_line_takes_first_nonempty_line() -> None:
    assert first_line("Preface\nThis guide...") == "Preface"


def test_first_line_strips_leading_blank_lines() -> None:
    assert first_line("\n\n  Title here \nbody") == "Title here"


def test_first_line_truncates_long_line() -> None:
    line = "x" * 200
    out = first_line(line)
    assert len(out) == _FIRST_LINE_MAX
    assert out.endswith("…")


def test_first_line_truncates_at_whitespace_boundary() -> None:
    # Space at the truncation point: rstrip drops it, so result is <= max_len.
    out = first_line("a " * 100)  # "a a a ..." spaces fall on the cut
    assert len(out) <= _FIRST_LINE_MAX
    assert out.endswith("…")
    assert "  " not in out.rstrip("…")


def test_first_line_keeps_short_line_verbatim() -> None:
    assert first_line("short") == "short"


def test_first_line_empty_content() -> None:
    assert first_line("") == ""
    assert first_line("   \n  ") == ""
