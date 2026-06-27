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
from pathlib import Path

import streamlit as st

from corpus_rag.prompts import ABSTENTION_ANSWER

logger = logging.getLogger(__name__)

_FIRST_LINE_MAX = 120

# File types the GUI accepts for upload → ingest. These are the Docling-supported
# formats in scope (root spec.md §3.1, §7.3); extensions only, no leading dot.
ALLOWED_UPLOAD_TYPES = ["pdf", "docx", "pptx", "html", "htm", "md"]

# Uploaded files are persisted here (by basename) rather than a random temp dir,
# so re-ingesting the same file resolves to the same path + provenance → the same
# content+meta-derived Document.id → OVERWRITE dedups instead of duplicating.
UPLOAD_DIR = Path("uploads")

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

    Each upload is persisted to ``UPLOAD_DIR`` under its basename (overwriting a
    prior copy) and ingested with a stable ``source`` meta (the basename). The
    stable path + stable meta make the Docling chunks' content+meta-derived
    Document.id deterministic, so re-ingesting the same file OVERWRITEs rather than
    duplicating (idempotent). New chunks are searchable on the next query.
    """
    from corpus_rag.pipelines.indexing import run_indexing

    pipeline = _get_ingest_pipeline()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    metas: list[dict] = []
    for up in uploaded_files:
        name = Path(up.name).name or "upload"  # basename only (no path traversal)
        dest = UPLOAD_DIR / name
        dest.write_bytes(up.getvalue())  # overwrite any prior copy of this name
        paths.append(str(dest))
        metas.append({"source": name})

    result = run_indexing(pipeline, paths, meta=metas)
    return result.get("writer", {}).get("documents_written", 0)


def _source_title(document) -> str:
    """A human title for a chunk: source filename → heading path → first line.

    Prefers the ``source`` meta we attach at ingest (the original filename), then
    Docling's doc-origin filename, then the chunk's heading trail, then its first
    line — so every row has a meaningful label, never "(unknown source)".
    """
    meta = document.meta or {}
    source = meta.get("source")
    if source:
        return str(source)
    dl = meta.get("dl_meta") or {}
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


# Display name for a stored chunk, as a SQL expression: our ingest-time ``source``
# meta, else Docling's doc-origin filename, else a placeholder. Shared by the list
# and the unload query so "remove X" deletes exactly the group shown as "X".
_SOURCE_NAME_SQL = (
    "COALESCE(meta->>'source', meta->'dl_meta'->'origin'->>'filename', '(unknown source)')"
)


@st.cache_data(ttl=30, show_spinner=False)
def _loaded_documents() -> list[tuple[str, int]]:
    """Distinct source documents in the store with chunk counts.

    Aggregates in SQL so the whole corpus (and its embeddings) is NOT pulled into
    the app on every Streamlit rerun — only ``(filename, count)`` rows come back.
    Cached briefly; ``_loaded_documents.clear()`` after an ingest/unload refreshes it.

    Assumes ``PgvectorDocumentStore``'s default table name (``haystack_documents``);
    this app exposes no table-name setting. Deriving it from a built store would
    force an embedding-model load (dimension contract) on every sidebar render.
    """
    import psycopg

    from corpus_rag.settings import get_settings

    sql = (
        f"SELECT {_SOURCE_NAME_SQL} AS name, COUNT(*) AS n "
        "FROM haystack_documents GROUP BY name ORDER BY name"
    )
    with psycopg.connect(get_settings().pg_conn_str) as conn:
        rows = conn.execute(sql).fetchall()
    return [(str(name), int(n)) for name, n in rows]


def _unload_document(name: str) -> int:
    """Delete every chunk whose display name == ``name``; return rows removed.

    Matches the exact expression the "Currently Loaded" list shows, so removing
    "report.pdf" deletes all of its chunks (including duplicate copies). Also
    deletes the persisted upload file so a later ``--reset`` rebuild won't re-add
    it. The retriever reads the store live, so the change applies to the next query.
    """
    import psycopg

    from corpus_rag.settings import get_settings

    sql = f"DELETE FROM haystack_documents WHERE {_SOURCE_NAME_SQL} = %s"
    with psycopg.connect(get_settings().pg_conn_str) as conn:
        removed = conn.execute(sql, (name,)).rowcount

    # Best-effort: drop the persisted upload of the same basename.
    upload = UPLOAD_DIR / Path(name).name
    if upload.is_file():
        upload.unlink()
    return int(removed)


def _render_ingest_sidebar() -> None:
    with st.sidebar:
        st.header("Documents")
        st.caption(f"Upload to ingest ({', '.join(ALLOWED_UPLOAD_TYPES)}).")

        # Surface the result of the just-completed ingest. It is stashed in
        # session_state because we rerun (to reset the uploader) right after.
        prior = st.session_state.pop("ingest_msg", None)
        if prior:
            level, text = prior
            (st.success if level == "ok" else st.error)(text)

        # Rotating key: bumping it after an ingest gives a fresh, empty uploader
        # on the next run, so the widget returns to its pre-upload state.
        round_ = st.session_state.setdefault("upload_round", 0)
        uploads = st.file_uploader(
            "Upload files",
            type=ALLOWED_UPLOAD_TYPES,
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"uploader_{round_}",
        )
        # Ingest immediately on upload — no separate button (it is implied).
        if uploads:
            total_bytes = sum(len(f.getvalue()) for f in uploads)
            if total_bytes > MAX_UPLOAD_BYTES:
                st.session_state["ingest_msg"] = (
                    "err",
                    f"Upload too large ({total_bytes // 1024 // 1024} MB); "
                    f"limit is {MAX_UPLOAD_MB} MB.",
                )
            else:
                try:
                    with st.spinner("Ingesting (convert → chunk → embed → store)…"):
                        written = _ingest_uploads(uploads)
                    _loaded_documents.clear()  # refresh the "Currently Loaded" list
                    st.session_state["ingest_msg"] = (
                        "ok",
                        f"Ingested {len(uploads)} file(s) → {written} chunk(s). Searchable now.",
                    )
                except Exception:  # noqa: BLE001 — generic message; detail to logs
                    logger.exception("Ingest failed")
                    st.session_state["ingest_msg"] = (
                        "err",
                        "Ingest failed. Check the server logs for details.",
                    )
            st.session_state["upload_round"] = round_ + 1  # reset the uploader
            st.rerun()

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
            label_col, btn_col = st.columns([0.82, 0.18])
            label_col.write(f"{name} — {n} chunk(s)")
            if btn_col.button("✕", key=f"unload_{name}", help=f"Remove {name} from the corpus"):
                try:
                    removed = _unload_document(name)
                    _loaded_documents.clear()
                    st.session_state["ingest_msg"] = (
                        "ok",
                        f"Removed {name} ({removed} chunk(s)).",
                    )
                except Exception:  # noqa: BLE001 — generic message; detail to logs
                    logger.exception("Unload failed")
                    st.session_state["ingest_msg"] = (
                        "err",
                        "Remove failed. Check the server logs for details.",
                    )
                st.rerun()


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
