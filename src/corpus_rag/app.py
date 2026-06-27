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


def _ingest_uploads(uploaded_files) -> int:
    """Ingest GUI-uploaded files into the corpus; return chunks written.

    Writes each upload to a private temp file (preserving its extension so Docling
    routes by format), runs the indexing pipeline into the same pgvector store the
    query path reads, then removes the temp files. New chunks are searchable on the
    next query (the retriever hits the DB live). Filenames are basename-only to
    avoid path traversal.
    """
    import shutil
    import tempfile
    from pathlib import Path

    from corpus_rag.document_store import build_document_store
    from corpus_rag.pipelines.indexing import build_indexing_pipeline, run_indexing
    from corpus_rag.settings import get_settings

    settings = get_settings()
    pipeline = build_indexing_pipeline(build_document_store(settings), settings)

    tmpdir = Path(tempfile.mkdtemp(prefix="corpus_upload_"))
    try:
        paths: list[str] = []
        for up in uploaded_files:
            dest = tmpdir / Path(up.name).name  # basename only (no traversal)
            dest.write_bytes(up.getvalue())
            paths.append(str(dest))
        result = run_indexing(pipeline, paths)
        return result.get("writer", {}).get("documents_written", 0)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _render_ingest_sidebar() -> None:
    with st.sidebar:
        st.header("Add documents")
        st.caption(
            "Upload files to ingest into the corpus "
            f"({', '.join(ALLOWED_UPLOAD_TYPES)}). "
            "Demo use only — synthetic / non-PHI documents."
        )
        uploads = st.file_uploader("Files", type=ALLOWED_UPLOAD_TYPES, accept_multiple_files=True)
        if uploads and st.button("Ingest uploaded files"):
            try:
                with st.spinner("Ingesting (convert → chunk → embed → store)…"):
                    written = _ingest_uploads(uploads)
                st.success(
                    f"Ingested {len(uploads)} file(s) → {written} chunk(s). "
                    "New content is searchable now."
                )
            except Exception:  # noqa: BLE001 — generic message; detail to logs
                logger.exception("Ingest failed")
                st.error("Ingest failed. Check the server logs for details.")


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

    # §2A.4: sources always co-rendered with the response.
    st.subheader(f"Sources ({len(sources)})")
    if not sources:
        st.info("No source chunks retrieved for this query.")
        return

    # Comparison table: rows in rerank order; show how rerank moved each chunk
    # away from its initial cosine rank (Δ = cosine_rank − rerank_rank; >0 = up).
    st.markdown(
        "**Reranking vs. cosine order** — rows in rerank order. "
        "`Δ` = how many places the cross-encoder moved a chunk up (+) or down (−) "
        "from its cosine rank."
    )
    st.dataframe(
        [
            {
                "Rerank #": s.rerank_rank,
                "Rerank score": round(s.rerank_score, 4) if s.rerank_score is not None else None,
                "Cosine #": s.cosine_rank,
                "Cosine score": round(s.cosine_score, 4) if s.cosine_score is not None else None,
                "Δ": s.cosine_rank - s.rerank_rank,
                "Source": first_line(s.document.content),
            }
            for s in sources
        ],
        use_container_width=True,
        hide_index=True,
    )

    for s in sources:
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


if __name__ == "__main__":
    main()
