"""Tests for the Streamlit app's pure helpers.

Only import-safe, side-effect-free logic is unit-tested here; the Streamlit
rendering path is exercised manually / via the live acceptance run.
"""

from __future__ import annotations

from haystack import Document

from corpus_rag.app import (
    _FIRST_LINE_MAX,
    ALLOWED_UPLOAD_TYPES,
    _answer_may_be_incomplete,
    _score_str,
    _source_title,
    first_line,
)


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
    # Space lands exactly at the slice point (index max_len-2): rstrip drops it,
    # so the result is strictly shorter than max_len.
    line = "x" * (_FIRST_LINE_MAX - 2) + " trailing" + "y" * 80
    out = first_line(line)
    assert len(out) < _FIRST_LINE_MAX  # rstrip consumed the boundary space
    assert out.endswith("…")
    assert not out[:-1].endswith(" ")


def test_first_line_keeps_short_line_verbatim() -> None:
    assert first_line("short") == "short"


def test_first_line_empty_content() -> None:
    assert first_line("") == ""
    assert first_line("   \n  ") == ""


def test_allowed_upload_types_are_bare_lowercase_extensions() -> None:
    # Streamlit file_uploader wants extensions without a leading dot.
    assert {"pdf", "docx", "html"} <= set(ALLOWED_UPLOAD_TYPES)
    assert all(t == t.lower() and not t.startswith(".") for t in ALLOWED_UPLOAD_TYPES)


def test_unload_document_deletes_chunks_and_removes_file(monkeypatch, tmp_path) -> None:
    """The only data-deletion path: DELETE by display name + remove the upload."""
    import psycopg

    from corpus_rag import app
    from corpus_rag import settings as settings_mod

    # Anchor uploads at a temp dir and seed the persisted file to be removed.
    monkeypatch.setattr(app, "UPLOAD_DIR", tmp_path)
    upload = tmp_path / "report.pdf"
    upload.write_bytes(b"pdf-bytes")

    captured: dict = {}

    class _FakeCursor:
        rowcount = 3

    class _FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, sql, params=None):
            captured["sql"] = sql
            captured["params"] = params
            return _FakeCursor()

    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: _FakeConn())
    monkeypatch.setattr(
        settings_mod,
        "get_settings",
        lambda: type("S", (), {"pg_conn_str": "postgresql://x"})(),
    )

    removed = app._unload_document("report.pdf")

    assert removed == 3
    assert "DELETE FROM haystack_documents" in captured["sql"]
    # Deletion keys on the SAME name expression the sidebar list shows.
    assert app._SOURCE_NAME_SQL in captured["sql"]
    assert captured["params"] == ("report.pdf",)
    assert not upload.exists()  # persisted upload removed too


def test_source_title_prefers_origin_filename() -> None:
    doc = Document(content="body", meta={"dl_meta": {"origin": {"filename": "guideline.pdf"}}})
    assert _source_title(doc) == "guideline.pdf"


def test_source_title_falls_back_to_heading_then_first_line() -> None:
    # No filename → most-specific heading.
    doc = Document(content="body", meta={"dl_meta": {"headings": ["Hand Hygiene"]}})
    assert _source_title(doc) == "Hand Hygiene"
    # No filename or heading → first line of the chunk.
    doc = Document(content="Vital Signs\nmore text", meta={})
    assert _source_title(doc) == "Vital Signs"


def test_source_title_never_blank() -> None:
    assert _source_title(Document(content="", meta={})) == "Untitled"


def test_score_str_formats_score() -> None:
    assert _score_str(0.9942) == "0.9942"
    assert _score_str(0.99417) == "0.9942"  # rounded to 4 dp
    assert _score_str(None) == "n/a"


def test_answer_may_be_incomplete_detects_token_limit_and_mid_sentence() -> None:
    # length = cap hit, still truncated.
    assert _answer_may_be_incomplete("Complete sentence.", "length")
    # No finish reason at all → fall back to mid-sentence heuristic.
    assert _answer_may_be_incomplete("Weight Management: Advise weight loss for", None)
    # Clean stop is TRUSTED even if it ends on a word (no false positive).
    assert not _answer_may_be_incomplete("Advise weight loss for the patient", "stop")
    assert not _answer_may_be_incomplete("Complete sentence.", "stop")
    # No-finish-reason but properly terminated → complete.
    assert not _answer_may_be_incomplete("Complete sentence.", None)
    # Any other reason (e.g. content_filter) is treated as incomplete.
    assert _answer_may_be_incomplete("Partial", "content_filter")
    assert not _answer_may_be_incomplete("Insufficient grounding in the corpus to answer.", None)


def test_ingest_uploads_reports_progress_and_batches(monkeypatch, tmp_path) -> None:
    """GUI ingest drives converter/embedder/writer directly for live progress."""
    from haystack import Document

    from corpus_rag import app

    monkeypatch.setattr(app, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(
        app,
        "get_settings",
        lambda: type("S", (), {"ingest_embed_batch_size": 2})(),
    )

    class _Upload:
        name = "guide.pdf"

        def getvalue(self):
            return b"pdf-bytes"

    class _Converter:
        def run(self, sources, meta):
            assert sources == [str(tmp_path / "guide.pdf")]
            assert meta == [{"source": "guide.pdf"}]
            return {
                "documents": [
                    Document(content="a", meta=meta[0]),
                    Document(content="b", meta=meta[0]),
                    Document(content="c", meta=meta[0]),
                ]
            }

    class _Embedder:
        def __init__(self):
            self.batch_sizes = []

        def run(self, documents):
            self.batch_sizes.append(len(documents))
            return {"documents": documents}

    class _Writer:
        def __init__(self):
            self.batch_sizes = []

        def run(self, documents):
            self.batch_sizes.append(len(documents))
            return {"documents_written": len(documents)}

    embedder = _Embedder()
    writer = _Writer()
    components = type(
        "Components",
        (),
        {"converter": _Converter(), "embedder": embedder, "writer": writer},
    )()
    monkeypatch.setattr(app, "_get_ingest_components", lambda: components)

    progress_events = []
    written = app._ingest_uploads([_Upload()], progress=lambda *args: progress_events.append(args))

    assert written == 3
    assert (tmp_path / "guide.pdf").read_bytes() == b"pdf-bytes"
    assert embedder.batch_sizes == [2, 1]
    assert writer.batch_sizes == [2, 1]
    assert progress_events[-1] == ("Indexed 3 of 3 chunks", 3, 3)
