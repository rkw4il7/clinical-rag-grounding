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

import streamlit as st

from corpus_rag.prompts import ABSTENTION_ANSWER

_FIRST_LINE_MAX = 120


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

    query = st.chat_input("Ask a question about the corpus")
    if not query:
        return

    # Render the query as plain text (avoid interpreting Markdown in user input).
    st.write("**Query:**", query)

    try:
        with st.spinner("Retrieving, reranking, and generating…"):
            answer, sources = run_query_reranked(query, engine=_get_engine())
    except Exception as exc:  # noqa: BLE001 — surface any failure in-app, not a traceback
        st.error(f"Query failed: {exc}")
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
