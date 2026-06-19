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


@st.cache_resource(show_spinner="Loading embedding model + vector store…")
def _get_pipeline():
    """Build the query pipeline once per process (embedder + store are heavy).

    Settings are captured at first call. Restart the Streamlit process to pick up
    changed EMBED_MODEL_ID, TOP_K, or LLM_* values from the environment / .env.
    """
    from corpus_rag.document_store import build_document_store
    from corpus_rag.pipelines.query import build_query_pipeline
    from corpus_rag.settings import get_settings

    settings = get_settings()
    return build_query_pipeline(build_document_store(settings), settings)


def main() -> None:
    from corpus_rag.pipelines.query import run_query

    st.set_page_config(page_title="Corpus RAG Explorer", layout="wide")
    st.title("Corpus RAG Explorer")
    st.caption(
        "Semantic retrieval + grounded generation. Answers are grounded in the "
        "retrieved corpus; specifics never come from the model's training data."
    )

    query = st.chat_input("Ask a question about the corpus")
    if not query:
        return

    # Render the query as plain text (avoid interpreting Markdown in user input).
    st.write("**Query:**", query)

    try:
        with st.spinner("Retrieving and generating…"):
            answer, documents = run_query(query, pipeline=_get_pipeline())
    except Exception as exc:  # noqa: BLE001 — surface any failure in-app, not a traceback
        st.error(f"Query failed: {exc}")
        return

    st.subheader("Response")
    if answer == ABSTENTION_ANSWER:
        st.warning(answer)
    else:
        st.markdown(answer)

    # §2A.4: sources always co-rendered with the response, in returned order.
    st.subheader(f"Sources ({len(documents)})")
    if not documents:
        st.info("No source chunks retrieved for this query.")
        return

    for i, doc in enumerate(documents, start=1):
        score = f"{doc.score:.4f}" if doc.score is not None else "n/a"
        with st.expander(f"{i}. {first_line(doc.content)}  ·  score {score}"):
            # Verbatim chunk text (the citation unit; byte-equal to the store).
            st.text(doc.content)
            st.markdown("**Provenance**")
            st.json(doc.meta or {})


if __name__ == "__main__":
    main()
