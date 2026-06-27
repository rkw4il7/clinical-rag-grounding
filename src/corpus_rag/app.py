"""Streamlit front end for the Corpus RAG Explorer (root ``spec.md`` §5).

Single page: a query box runs the grounded query pipeline and renders the query,
the generated response, and — in rerank order — the chunks that actually grounded
it (at/above the MIN_SCORE floor) as a single table: ordinal/numeric columns
(rerank #, cosine #, Δ, scores) on the left and the **verbatim** chunk text on the
right. Below-threshold candidates that did not contribute are not shown, and no
internal metadata structure is dumped to the page.

§2A.4 co-rendering: the response is never shown without its ranked, verbatim
ground-truth sources beside it; on abstention no fabricated clinical claim is
shown.

Run:  uv run streamlit run src/corpus_rag/app.py
"""

from __future__ import annotations

import logging

import streamlit as st

from corpus_rag.prompts import ABSTENTION_ANSWER

logger = logging.getLogger(__name__)

_FIRST_LINE_MAX = 120

# File types the GUI accepts for upload → ingest. These are the Docling-supported
# formats in scope (root spec.md §3.1, §7.3); extensions only, no leading dot.
ALLOWED_UPLOAD_TYPES = ["pdf", "docx", "pptx", "html", "htm", "md"]

# Cap total upload size: ingest (esp. OCR, on by default) runs in the Streamlit
# request handler, so a huge upload would block the worker for minutes/hours.
MAX_UPLOAD_MB = 50
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


def first_line(content: str, max_len: int = _FIRST_LINE_MAX) -> str:
    """Derive the expander label from a chunk's content (root §4.4).

    First non-empty line of ``content``, truncated to ``max_len`` chars.
    """
    stripped = content.strip()
    line = stripped.splitlines()[0].rstrip() if stripped else ""
    if len(line) > max_len:
        return line[: max_len - 1].rstrip() + "…"
    return line


@st.cache_resource(show_spinner="Loading embedding + rerank models + vector store…")
def _get_engine():
    """Build the rerank query engine once per process (models + store are heavy).

    Settings are captured at first call. Restart the Streamlit process to pick up
    changed EMBED_MODEL_ID, RERANK_*, or LLM_* values from the environment / .env.
    """
    from corpus_rag.document_store import build_document_store
    from corpus_rag.pipelines.query import build_rerank_engine
    from corpus_rag.settings import get_settings

    settings = get_settings()
    return build_rerank_engine(build_document_store(settings), settings)


@st.cache_resource(show_spinner="Loading ingest pipeline…")
def _get_ingest_pipeline():
    """Build the Docling → embed → write pipeline once (the embedder is heavy)."""
    from corpus_rag.document_store import build_document_store
    from corpus_rag.pipelines.indexing import build_indexing_pipeline
    from corpus_rag.settings import get_settings

    settings = get_settings()
    return build_indexing_pipeline(build_document_store(settings), settings)


def _ingest_uploads(uploaded_files) -> int:
    """Ingest GUI-uploaded files into the corpus; return chunks written.

    Writes each upload to a private temp file (extension preserved so Docling
    routes by format), runs the cached indexing pipeline into the same pgvector
    store the query path reads, then removes the temp files. New chunks are
    searchable on the next query. Temp names are basename-only (no path traversal)
    and index-prefixed so two uploads sharing a basename can't clobber each other.
    """
    import shutil
    import tempfile
    from pathlib import Path

    from corpus_rag.pipelines.indexing import run_indexing

    pipeline = _get_ingest_pipeline()
    tmpdir = Path(tempfile.mkdtemp(prefix="corpus_upload_"))
    try:
        paths: list[str] = []
        for i, up in enumerate(uploaded_files):
            safe = Path(up.name).name or f"upload_{i}"  # basename; fallback if empty
            dest = tmpdir / f"{i}_{safe}"  # index prefix avoids basename collisions
            dest.write_bytes(up.getvalue())
            paths.append(str(dest))
        result = run_indexing(pipeline, paths)
        return result.get("writer", {}).get("documents_written", 0)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _source_title(document) -> str:
    """A human title for a chunk: filename → heading path → first line of text.

    Never returns a placeholder like "(unknown source)" — falls back through the
    Docling provenance (origin filename, then heading trail) to the chunk's own
    first line so every row has a meaningful label.
    """
    dl = (document.meta or {}).get("dl_meta") or {}
    origin = dl.get("origin") or {}
    name = origin.get("filename")
    if name:
        return str(name)
    headings = dl.get("headings") or []
    if headings:
        return str(headings[-1])
    return first_line(document.content) or "Untitled"


def _rank_score(rank: int, score: float | None) -> str:
    """Combine an ordinal rank and its score as ``1 / 0.9942`` (or ``1 / n/a``)."""
    return f"{rank} / {score:.4f}" if score is not None else f"{rank} / n/a"


@st.cache_data(ttl=30, show_spinner=False)
def _loaded_documents() -> list[tuple[str, int]]:
    """Distinct source documents in the store with chunk counts.

    Aggregates in SQL so the whole corpus (and its embeddings) is NOT pulled into
    the app on every Streamlit rerun — only ``(filename, count)`` rows come back.
    Cached briefly; ``_loaded_documents.clear()`` after an ingest refreshes it.

    Assumes ``PgvectorDocumentStore``'s default table name (``haystack_documents``);
    this app exposes no table-name setting. Deriving it from a built store would
    force an embedding-model load (dimension contract) on every sidebar render.
    """
    import psycopg

    from corpus_rag.settings import get_settings

    sql = (
        "SELECT COALESCE(meta->'dl_meta'->'origin'->>'filename', '(unknown source)') "
        "AS name, COUNT(*) AS n FROM haystack_documents GROUP BY name ORDER BY name"
    )
    with psycopg.connect(get_settings().pg_conn_str) as conn:
        rows = conn.execute(sql).fetchall()
    return [(str(name), int(n)) for name, n in rows]


def _render_ingest_sidebar() -> None:
    with st.sidebar:
        st.header("Documents")
        st.caption(f"Upload to ingest ({', '.join(ALLOWED_UPLOAD_TYPES)}).")
        uploads = st.file_uploader(
            "Upload files",
            type=ALLOWED_UPLOAD_TYPES,
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploads and st.button("Ingest uploaded files"):
            total_bytes = sum(len(f.getvalue()) for f in uploads)
            if total_bytes > MAX_UPLOAD_BYTES:
                st.error(
                    f"Upload too large ({total_bytes // 1024 // 1024} MB); "
                    f"limit is {MAX_UPLOAD_MB} MB."
                )
                return
            try:
                with st.spinner("Ingesting (convert → chunk → embed → store)…"):
                    written = _ingest_uploads(uploads)
                _loaded_documents.clear()  # refresh the "Currently Loaded" list
                st.success(
                    f"Ingested {len(uploads)} file(s) → {written} chunk(s). "
                    "New content is searchable now."
                )
            except Exception:  # noqa: BLE001 — generic message; detail to logs
                logger.exception("Ingest failed")
                st.error("Ingest failed. Check the server logs for details.")

        st.subheader("Currently Loaded")
        try:
            loaded = _loaded_documents()
        except Exception:  # noqa: BLE001 — store may be down; don't crash the page
            logger.exception("Listing loaded documents failed")
            st.caption("Could not list documents (store unavailable).")
            return
        if not loaded:
            st.caption("No documents ingested yet.")
            return
        for name, n in loaded:
            st.write(f"- {name} — {n} chunk(s)")


def main() -> None:
    from corpus_rag.pipelines.query import run_query_reranked

    st.set_page_config(page_title="Corpus RAG Explorer", layout="wide")
    st.title("Corpus RAG Explorer")

    _render_ingest_sidebar()

    query = st.chat_input("Ask a question about the corpus")
    if not query:
        return

    # Render the query as plain text (avoid interpreting Markdown in user input).
    st.write("**Query:**", query)

    try:
        with st.spinner("Retrieving, reranking, and generating…"):
            answer, sources = run_query_reranked(query, engine=_get_engine())
    except Exception:  # noqa: BLE001 — show a generic message; detail goes to logs
        # Never surface raw exception text in a clinical-facing UI: DB/LLM errors
        # can embed the connection string (credentials) or other internals.
        logger.exception("Query failed")
        st.error("Query failed. Check the server logs for details.")
        return

    st.subheader("Response")
    if answer == ABSTENTION_ANSWER:
        st.warning(answer)
    else:
        st.markdown(answer)

    # Show exactly the chunks the generator was fed (the pipeline marks them);
    # below-floor and beyond-top-K candidates did not contribute, so they are
    # not displayed.
    grounded = [s for s in sources if s.used_for_grounding]

    st.subheader("Sources")
    if not grounded:
        st.info(
            "No retrieved chunk met the MIN_SCORE grounding floor — the response above abstains."
        )
        return

    # §2A.4: grounded sources co-rendered with the response, in rerank order.
    # One table — Source title left, rank/score columns, verbatim chunk text right.
    # "Rank" = rerank rank/score; "Similarity" = cosine rank/score (both "n / s").
    st.dataframe(
        [
            {
                "Source": _source_title(s.document),
                "Rank": _rank_score(s.rerank_rank, s.rerank_score),
                "Similarity": _rank_score(s.cosine_rank, s.cosine_score),
                "Chunk text": s.document.content,
            }
            for s in grounded
        ],
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
