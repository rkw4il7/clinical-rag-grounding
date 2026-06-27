"""Streamlit front end for the Corpus RAG Explorer (root ``spec.md`` §5).

Single page: a query box runs the grounded query pipeline and renders the query,
the generated response, and the top-K retrieved source chunks — each as a
click-to-expand row showing the **verbatim** chunk content, its provenance
metadata, and similarity score, in the retriever's returned (cosine-ranked)
order.

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


def _source_name(meta: dict | None) -> str:
    """Best-effort source filename from a chunk's Docling provenance metadata."""
    dl = (meta or {}).get("dl_meta") or {}
    origin = dl.get("origin") or {}
    return origin.get("filename") or "(unknown source)"


@st.cache_data(ttl=30, show_spinner=False)
def _loaded_documents() -> list[tuple[str, int]]:
    """Distinct source documents in the store with chunk counts.

    Aggregates in SQL so the whole corpus (and its embeddings) is NOT pulled into
    the app on every Streamlit rerun — only ``(filename, count)`` rows come back.
    Cached briefly; ``_loaded_documents.clear()`` after an ingest refreshes it.
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


def _fmt(score: float | None) -> str:
    return f"{score:.4f}" if score is not None else "n/a"


def main() -> None:
    from corpus_rag.pipelines.query import run_query_reranked

    st.set_page_config(page_title="Corpus RAG Explorer", layout="wide")
    st.title("Corpus RAG Explorer")
    st.caption(
        "Semantic retrieval + cross-encoder reranking + grounded generation. "
        "Answers are grounded in the retrieved corpus; specifics never come from "
        "the model's training data."
    )

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

    if not sources:
        st.subheader("Sources (0)")
        st.info("No source chunks retrieved for this query.")
        return

    # Split by the grounding floor so the UI never implies a below-threshold chunk
    # grounded the answer. Only chunks at/above MIN_SCORE are "Sources used"; the
    # rest are shown separately as retrieval candidates that were NOT used.
    from corpus_rag.settings import get_settings

    floor = get_settings().min_score

    def _is_grounded(rs) -> bool:
        return floor <= 0.0 or (rs.cosine_score or 0.0) >= floor

    grounded = [s for s in sources if _is_grounded(s)]
    below = [s for s in sources if not _is_grounded(s)]

    # Retrieval diagnostics over ALL candidates (how rerank moved each vs cosine).
    st.markdown(
        "**Retrieval diagnostics — all candidates** (rerank order). "
        "`Δ` = places the cross-encoder moved a chunk up (+) / down (−) from its "
        "cosine rank. `Grounded` = at/above the MIN_SCORE floor."
    )
    st.dataframe(
        [
            {
                "Rerank #": s.rerank_rank,
                "Rerank score": round(s.rerank_score, 4) if s.rerank_score is not None else None,
                "Cosine #": s.cosine_rank,
                "Cosine score": round(s.cosine_score, 4) if s.cosine_score is not None else None,
                "Δ": s.cosine_rank - s.rerank_rank,
                "Grounded": _is_grounded(s),
                "Source": first_line(s.document.content),
            }
            for s in sources
        ],
        use_container_width=True,
        hide_index=True,
    )

    # §2A.4: the grounded sources are co-rendered with the response.
    st.subheader(f"Sources used for grounding ({len(grounded)})")
    if not grounded:
        st.info(
            "No retrieved chunk met the MIN_SCORE grounding floor — the response "
            "above abstains. Below-threshold candidates are listed separately."
        )
    for s in grounded:
        label = (
            f"#{s.rerank_rank} (cosine #{s.cosine_rank})  ·  "
            f"rerank {_fmt(s.rerank_score)}  ·  cosine {_fmt(s.cosine_score)}  ·  "
            f"{first_line(s.document.content)}"
        )
        with st.expander(label):
            # Verbatim chunk text (the citation unit; byte-equal to the store).
            st.text(s.document.content)
            st.markdown("**Provenance**")
            st.json(s.document.meta or {})

    if below:
        with st.expander(
            f"Other retrieved candidates below MIN_SCORE — NOT used for grounding ({len(below)})"
        ):
            for s in below:
                st.markdown(f"- cosine {_fmt(s.cosine_score)} · {first_line(s.document.content)}")


if __name__ == "__main__":
    main()
