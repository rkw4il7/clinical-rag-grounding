"""Streamlit front end for the Grounded RAG Explorer (root ``spec.md`` §5).

A query box runs the grounded query pipeline; the answer and its sources render in
two tabs. The **Sources** tab lists — in rerank order — the chunks that actually
grounded the answer (at/above the MIN_SCORE floor): the source title, the rerank
and cosine scores, and the **verbatim** chunk text. Below-threshold candidates
that did not contribute are not shown, and no internal metadata is dumped to the
page.

§2A.4 co-rendering: the response is never shown without its ranked, verbatim
ground-truth sources alongside it; on abstention no fabricated clinical claim is
shown.

Run:  uv run streamlit run src/corpus_rag/app.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from corpus_rag.logging_setup import quiet_noisy_upstream
from corpus_rag.prompts import ABSTENTION_ANSWER
from corpus_rag.settings import get_settings

logger = logging.getLogger(__name__)

_FIRST_LINE_MAX = 120

# File types the GUI accepts for upload → ingest. These are the Docling-supported
# formats in scope (root spec.md §3.1, §7.3); extensions only, no leading dot.
ALLOWED_UPLOAD_TYPES = ["pdf", "docx", "pptx", "html", "htm", "md"]

# Uploaded files are persisted here (by basename) rather than a random temp dir,
# so re-ingesting the same file resolves to the same path + provenance → the same
# content+meta-derived Document.id → OVERWRITE dedups instead of duplicating.
# Anchored to the PROJECT ROOT (not CWD) so idempotency holds regardless of where
# the Streamlit process was launched from. (app.py is src/corpus_rag/app.py.)
UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "uploads"

# Sidebar document-panel styling (targets Streamlit internal data-testids):
# - Hide the drop zone's repeated accepted-types/size helper line; the input,
#   drag/drop, and type validation remain intact.
# - Center the drop zone's "Browse files" button horizontally.
# - Center the "Delete selected" button (keyed container) horizontally.
_UPLOAD_ZONE_CSS = """
<style>
.st-key-doc-upload [data-testid="stFileUploaderDropzoneInstructions"] {
    display: none;
}
.st-key-doc-upload section[data-testid="stFileUploaderDropzone"] {
    justify-content: center;
}
</style>
"""

# Restyle st.tabs from Streamlit's minimal underline into a classic Windows/Mac
# "folder tab" set: bordered tabs with rounded tops sitting on the content pane,
# the active tab merged into the pane below it. Targets BaseWeb's internal
# data-baseweb hooks — revisit on a Streamlit/BaseWeb upgrade. Dark blue (#00008b)
# matches the Sources table borders for a consistent look.
_TABSET_CSS = """
<style>
/* The tab strip sits on a single bottom line that doubles as the pane's top edge. */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid #00008b;
}
/* Kill BaseWeb's default sliding underline + border so only our folders show. */
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    background-color: transparent;
    height: 0;
}
/* Each tab: a bordered folder with rounded top corners, overlapping the strip line. */
.stTabs [data-baseweb="tab"] {
    height: auto;
    padding: 6px 16px;
    margin-bottom: -1px;
    background: #f0f2f6;
    border: 1px solid #00008b;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
}
/* Active tab: white fill, its bottom edge erased so it connects to the pane. */
.stTabs [aria-selected="true"] {
    background: #ffffff;
    border-bottom: 1px solid #ffffff;
    font-weight: 600;
}
/* The content pane: a box under the tabs (top edge = the strip's bottom line). */
.stTabs [data-baseweb="tab-panel"] {
    border: 1px solid #00008b;
    border-top: none;
    border-radius: 0 0 6px 6px;
    padding: 12px 16px;
}
</style>
"""

# Hide the bottom-pinned chat input while a query is in flight. Emitted only on
# the busy run; a scoped display:none reliably removes the rendered widget (which
# st.empty()/pruning cannot, since chat_input self-pins to the app bottom chrome).
_HIDE_CHAT_INPUT_CSS = (
    '<style>[data-testid="stChatInput"], '
    '[data-testid="stChatInputContainer"] { display: none !important; }</style>'
)

# Push the document-management sidebar content down from the very top so it sits
# lower in the pane. The sidebar is left-docked, so horizontal centering is not
# possible; this is the vertical offset. Targets the sidebar content wrappers
# across Streamlit versions (internal data-testids).
_SIDEBAR_OFFSET_CSS = """
<style>
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 8vh;
}
</style>
"""

_COMPLETE_ANSWER_ENDINGS = tuple(".!?:;)]}\"'")

# Cap the Sources table to this many visible rows; past it, st.dataframe scrolls
# (a fixed pixel height is the only lever — there is no row-count parameter).
_SOURCES_TABLE_MAX_ROWS = 15
_DATAFRAME_ROW_PX = 35  # approx Streamlit row height; +1 for the header row


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

    settings = get_settings()
    return build_rerank_engine(build_document_store(settings), settings)


@st.cache_resource(show_spinner="Loading ingest models + vector store…")
def _get_ingest_components():
    """Build and warm the Docling → embed → write components once per process."""
    from corpus_rag.document_store import build_document_store
    from corpus_rag.pipelines.indexing import build_indexing_components

    settings = get_settings()
    components = build_indexing_components(build_document_store(settings), settings)
    components.embedder.warm_up()
    return components


@st.cache_resource(show_spinner="Preparing vector store…")
def _ensure_store_ready() -> bool:
    """Create the store table + HNSW index on a fresh database, once per process.

    The app must run against an empty DB (e.g. a just-started container) with no
    prior ingest. ``build_document_store`` constructs ``PgvectorDocumentStore``,
    which creates the table + index idempotently (and asserts the §7.2 dimension
    contract). Done at startup so the corpus is queryable/ingestable immediately —
    the table is never "missing". Returns True on success; the caller renders a
    soft error if the store is unreachable.
    """
    from corpus_rag.document_store import build_document_store

    build_document_store(get_settings())
    return True


def _ingest_uploads(uploaded_files, progress=None) -> int:
    """Ingest GUI-uploaded files into the corpus; return chunks written.

    Each upload is persisted to ``UPLOAD_DIR`` under its basename (overwriting a
    prior copy) and ingested with a stable ``source`` meta (the basename). The
    stable path + stable meta make the Docling chunks' content+meta-derived
    Document.id deterministic, so re-ingesting the same file OVERWRITEs rather than
    duplicating (idempotent). New chunks are searchable on the next query.
    """
    components = _get_ingest_components()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    def report(message: str, done: int = 0, total: int = 0) -> None:
        if progress is not None:
            progress(message, done, total)

    saved: list[tuple[str, Path, dict]] = []
    for up in uploaded_files:
        name = Path(up.name).name or "upload"  # basename only (no path traversal)
        dest = UPLOAD_DIR / name
        report(f"Saving {name}")
        dest.write_bytes(up.getvalue())  # overwrite any prior copy of this name
        saved.append((name, dest, {"source": name}))

    documents = []
    for i, (name, path, meta) in enumerate(saved, start=1):
        report(f"Converting {name} ({i}/{len(saved)})")
        converted = components.converter.run(sources=[str(path)], meta=[meta])["documents"]
        documents.extend(converted)
        report(f"Converted {name}: {len(converted)} chunk(s)", 0, len(documents))

    total = len(documents)
    if total == 0:
        report("No chunks produced from upload", 0, 0)
        return 0

    written = 0
    batch_size = get_settings().ingest_embed_batch_size
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        report(f"Embedding chunks {start + 1}-{end} of {total}", start, total)
        embedded = components.embedder.run(documents=documents[start:end])["documents"]
        report(f"Writing chunks {start + 1}-{end} of {total}", start, total)
        result = components.writer.run(documents=embedded)
        written += result.get("documents_written", 0)
        report(f"Indexed {end} of {total} chunks", end, total)

    return written


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


def _score_str(score: float | None) -> str:
    """Format a score value as ``0.9942`` (or ``n/a`` when absent)."""
    return f"{score:.4f}" if score is not None else "n/a"


def _answer_may_be_incomplete(answer: str, finish_reason: str | None) -> bool:
    """Detect a truncated answer the auto-continuation could not finish.

    ``finish_reason == "length"`` means the round cap was hit (still truncated).
    A clean ``"stop"`` is TRUSTED — the model chose to end, so no punctuation
    second-guessing (that produced false positives on answers ending in a word).
    Only when the backend reports no finish reason at all do we fall back to the
    mid-sentence heuristic, since such backends never signal length truncation.
    """
    stripped = answer.rstrip()
    if not stripped or stripped == ABSTENTION_ANSWER:
        return False
    if finish_reason == "length":
        return True
    if finish_reason == "stop":
        return False
    if finish_reason is not None:
        return True
    return not stripped.endswith(_COMPLETE_ANSWER_ENDINGS)


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

    sql = (
        f"SELECT {_SOURCE_NAME_SQL} AS name, COUNT(*) AS n "
        "FROM haystack_documents GROUP BY name ORDER BY name"
    )
    try:
        with psycopg.connect(get_settings().pg_conn_str) as conn:
            rows = conn.execute(sql).fetchall()
    except psycopg.errors.UndefinedTable:
        # Fresh database: the store table is created on first ingest. No table yet
        # simply means no documents are loaded — not an error.
        return []
    return [(str(name), int(n)) for name, n in rows]


def _corpus_has_documents() -> bool:
    """True if the store holds at least one chunk.

    Plain query with NO Streamlit calls, so it is safe to run BEFORE
    ``st.set_page_config`` to choose the initial sidebar state. A missing table
    (fresh DB) or an unreachable store both count as "empty".
    """
    import psycopg

    try:
        with psycopg.connect(get_settings().pg_conn_str) as conn:
            row = conn.execute("SELECT EXISTS (SELECT 1 FROM haystack_documents)").fetchone()
    except psycopg.errors.UndefinedTable:
        return False
    except Exception:  # noqa: BLE001 — store unreachable: treat as empty, don't crash
        logger.exception("Corpus presence check failed")
        return False
    return bool(row and row[0])


def _unload_document(name: str) -> int:
    """Delete every chunk whose display name == ``name``; return rows removed.

    Matches the exact expression the "Currently Loaded" list shows, so removing
    "report.pdf" deletes all of its chunks (including duplicate copies). Also
    deletes the persisted upload file so a later ``--reset`` rebuild won't re-add
    it. The retriever reads the store live, so the change applies to the next query.
    """
    import psycopg

    sql = f"DELETE FROM haystack_documents WHERE {_SOURCE_NAME_SQL} = %s"
    with psycopg.connect(get_settings().pg_conn_str) as conn:
        removed = conn.execute(sql, (name,)).rowcount

    # Best-effort: drop the persisted upload of the same basename.
    upload = UPLOAD_DIR / Path(name).name
    if upload.is_file():
        upload.unlink()
    return int(removed)


@st.fragment
def _render_ingest_sidebar() -> None:
    # A fragment so submitting a query (which runs the slow retrieve→rerank→
    # generate in the main area) does NOT re-execute/re-stream this section — that
    # re-stream was rendering the "Sources Currently Loaded" table a second time
    # while generation was in flight. Upload/unload below call st.rerun() for a
    # full refresh when the corpus actually changes.
    #
    # NOTE: fragments cannot call st.sidebar themselves — the caller wraps this in
    # `with st.sidebar:` (see main()).
    # Whether this whole sidebar is shown or collapsed is decided in main() via
    # st.set_page_config(initial_sidebar_state=...): collapsed when the corpus
    # already has documents (open on the RAG explorer), expanded when it is empty
    # (the user must add documents first). The native sidebar chevron reopens it.
    cap_mb = get_settings().upload_max_mb
    cap_bytes = cap_mb * 1024 * 1024

    st.markdown(_SIDEBAR_OFFSET_CSS, unsafe_allow_html=True)
    st.header("Manage Documents")
    st.caption(
        f"Upload to ingest ({', '.join(ALLOWED_UPLOAD_TYPES)}). File Size Limit: {cap_mb} MB"
    )

    # Surface the result of the just-completed ingest. It is stashed in
    # session_state because we rerun (to reset the uploader) right after.
    prior = st.session_state.pop("ingest_msg", None)
    if prior:
        level, text = prior
        (st.success if level == "ok" else st.error)(text)

    # Rotating key: bumping it after an ingest gives a fresh, empty uploader
    # on the next run, so the widget returns to its pre-upload state.
    round_ = st.session_state.setdefault("upload_round", 0)
    st.markdown(_UPLOAD_ZONE_CSS, unsafe_allow_html=True)
    # Render the label as a subheader (matching "Sources Currently Loaded") and
    # collapse the uploader's own small label so the heading is the only one.
    st.subheader("Drag Files Here...")
    with st.container(key="doc-upload"):
        uploads = st.file_uploader(
            "Drag Files Here...",
            type=ALLOWED_UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"uploader_{round_}",
            label_visibility="collapsed",
        )
    # Ingest immediately on upload — no separate button (it is implied).
    if uploads:
        total_bytes = sum(len(f.getvalue()) for f in uploads)
        if total_bytes > cap_bytes:
            st.session_state["ingest_msg"] = (
                "err",
                f"Upload too large ({total_bytes // 1024 // 1024} MB); limit is {cap_mb} MB.",
            )
        else:
            ingest_status = None
            try:
                # Warm the cached ingest models FIRST so the @st.cache_resource
                # load spinner fires alone, not stacked under our status box (same
                # pattern as the query path). Cached, so the call inside
                # _ingest_uploads is then a no-op.
                _get_ingest_components()

                # ONE status box; the progress bar is nested INSIDE it (not a
                # separate top-level element) so the user sees a single indicator.
                ingest_status = st.status("Ingesting upload...", expanded=True)
                ingest_progress = ingest_status.progress(0, text="Starting ingest")

                def show_ingest_progress(message: str, done: int = 0, total: int = 0) -> None:
                    if total > 0:
                        ingest_progress.progress(
                            min(done / total, 1.0),
                            text=f"{message} ({done}/{total} chunks)",
                        )
                    else:
                        ingest_progress.progress(0, text=message)

                written = _ingest_uploads(uploads, progress=show_ingest_progress)
                ingest_progress.progress(1.0, text=f"Indexed {written} chunk(s)")
                ingest_status.update(label="Ingest complete", state="complete", expanded=False)
                _loaded_documents.clear()  # refresh the "Currently Loaded" list
                st.session_state["ingest_msg"] = (
                    "ok",
                    f"Ingested {len(uploads)} file(s) → {written} chunk(s). Searchable now.",
                )
            except Exception:  # noqa: BLE001 — generic message; detail to logs
                logger.exception("Ingest failed")
                # Settle the spinner immediately (if the status box was created)
                # rather than leaving it animated until the rerun.
                if ingest_status is not None:
                    ingest_status.update(label="Ingest failed", state="error")
                st.session_state["ingest_msg"] = (
                    "err",
                    "Ingest failed. Check the server logs for details.",
                )
        st.session_state["upload_round"] = round_ + 1  # reset the uploader
        st.rerun()

    st.subheader("Sources Currently Loaded")
    # Create the store table on a fresh DB so the corpus is usable with zero prior
    # ingest; tolerate the store being unreachable.
    try:
        _ensure_store_ready()
        loaded = _loaded_documents()
    except Exception:  # noqa: BLE001 — store may be down; don't crash the page
        logger.exception("Listing loaded documents failed")
        st.caption("Could not reach the vector store.")
        return
    if not loaded:
        st.caption("No documents ingested yet.")
        return

    table_rows = [{"Source": name, "Chunks": n} for name, n in loaded]
    selection = st.dataframe(
        table_rows,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Source": st.column_config.TextColumn("Source"),
            "Chunks": st.column_config.NumberColumn("Chunks", format="%d"),
        },
    )
    selected_rows = selection.selection.rows
    selected_name = table_rows[selected_rows[0]]["Source"] if selected_rows else None
    button_label = f"Delete selected: {selected_name}" if selected_name else "Delete selected"
    # Center via a 3-column "pane": the button fills the wider middle column, which
    # is itself centered, so it reads as a centered block (reliable, no CSS).
    _, mid, _ = st.columns([1, 2, 1])
    delete_clicked = mid.button(
        button_label,
        disabled=selected_name is None,
        type="secondary",
        width="stretch",
    )
    if delete_clicked:
        try:
            removed = _unload_document(selected_name)
            _loaded_documents.clear()
            st.session_state["ingest_msg"] = (
                "ok",
                f"Removed {selected_name} ({removed} chunk(s)).",
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

    # Quiet benign upstream tokenizer logs — only when the app actually runs, not
    # on import (so tests that import this module keep their warnings intact).
    quiet_noisy_upstream()

    # Collapse the document-management sidebar by default once the corpus has
    # documents (open on the RAG explorer); expand it when empty so the user adds
    # documents first. This is the only programmatic sidebar control Streamlit
    # offers — the initial state per load; the user can reopen via the chevron.
    # _corpus_has_documents() makes no Streamlit calls, so it is safe before
    # set_page_config (which must be the first Streamlit command). Deliberately
    # NOT @st.cache_data: that machinery before set_page_config risks the
    # "first command" rule and adds an invalidation burden, and a single
    # SELECT EXISTS per rerun is negligible for this single-user app.
    sidebar_state = "collapsed" if _corpus_has_documents() else "expanded"
    st.set_page_config(
        page_title="Grounded RAG Explorer",
        layout="wide",
        initial_sidebar_state=sidebar_state,
    )
    st.title("Grounded RAG Explorer")

    # Fragments cannot call st.sidebar internally, so put it on the sidebar here.
    with st.sidebar:
        _render_ingest_sidebar()

    # Submitting stashes the query and reruns; the busy run then processes it. The
    # chat input self-pins to the app's bottom chrome, so st.empty()/pruning cannot
    # remove it mid-run — instead the busy run injects a scoped display:none that
    # hides the rendered widget for the whole retrieve→rerank→generate wait. The CSS
    # is not re-emitted on the post-processing rerun, so the input returns then.
    pending = st.session_state.get("pending_query")
    if pending is None:
        typed = st.chat_input("Ask a question...")
        if typed:
            st.session_state["pending_query"] = typed
            st.rerun()
    else:
        st.markdown(_HIDE_CHAT_INPUT_CSS, unsafe_allow_html=True)

        # Resolve the engine FIRST, OUTSIDE the status placeholder. On a fresh
        # process the @st.cache_resource model-load spinner fires here; doing it
        # before the status line means that spinner and our status never stack as
        # two status lines (the original "two status lines" bug). Cached after the
        # first query, so this is a no-op spinner thereafter.
        engine = _get_engine()

        # ONE status component. The step messages drive its label; the streamed
        # tokens render into a single placeholder NESTED INSIDE the same status box
        # (not a second widget). So the user sees live streaming and a single
        # status — both fed here — and the whole box is cleared at the end, with the
        # finished answer rendered in the Response tab below.
        status_area = st.empty()
        try:
            status = status_area.status("Starting RAG query…", expanded=True)
            answer_preview = status.empty()
            streamed: list[str] = []
            finish_reasons: list[str | None] = []

            def show_query_progress(message: str) -> None:
                status.update(label=message)

            def show_generation_progress(text: str) -> None:
                if text:
                    streamed.append(text)
                    answer_preview.markdown("".join(streamed))

            answer, sources = run_query_reranked(
                pending,
                engine=engine,
                progress=show_query_progress,
                generation_progress=show_generation_progress,
                finish_reason_callback=finish_reasons.append,
            )
            status_area.empty()
            st.session_state["rag_result"] = {
                "query": pending,
                "answer": answer,
                "sources": sources,
                "finish_reason": finish_reasons[0] if finish_reasons else None,
            }
        except Exception:  # noqa: BLE001 — show a generic message; detail goes to logs
            # Never surface raw exception text in a clinical-facing UI: DB/LLM errors
            # can embed the connection string (credentials) or other internals.
            logger.exception("Query failed")
            status_area.empty()
            st.session_state["pending_query"] = None
            st.error("Query failed. Check the server logs for details.")
            return
        # Rerun so the input returns (and the freshly stored result renders below).
        st.session_state["pending_query"] = None
        st.rerun()

    result = st.session_state.get("rag_result")
    if not result:
        return

    query = result["query"]
    answer = result["answer"]
    sources = result["sources"]
    finish_reason = result.get("finish_reason")

    # "Query" as a subheader (parity with the tab labels) with the question on the
    # next line, rendered as plain text so Markdown in user input is not interpreted.
    st.subheader("Query")
    st.text(query)

    # §2A.4: the grounded chunks the generator was actually fed (the pipeline marks
    # them); below-floor / beyond-top-K candidates did not contribute → not shown.
    grounded = [s for s in sources if s.used_for_grounding]

    st.markdown(_TABSET_CSS, unsafe_allow_html=True)
    tab_response, tab_sources = st.tabs(["Response", "Sources"])

    with tab_response:
        # Length-truncated answers are continued automatically inside the query
        # pipeline (transparent). If one is STILL incomplete here, the continuation
        # cap was hit — surface a passive note, no manual action.
        if _answer_may_be_incomplete(answer, finish_reason):
            st.warning(
                "The response may still be incomplete after automatic continuation. "
                "Increase LLM_MAX_TOKENS or MAX_CONTINUATION_ROUNDS for longer answers."
            )
        if answer == ABSTENTION_ANSWER:
            st.warning(answer)
        else:
            st.markdown(answer)

    with tab_sources:
        if not grounded:
            st.info("No retrieved chunk met the MIN_SCORE grounding floor — the response abstains.")
        else:
            # Grounded sources ordered by rerank score (highest first). One table:
            # Source title, score columns, then verbatim chunk text. "ReRank" =
            # cross-encoder score; "Similarity" = cosine.
            grounded = sorted(
                grounded,
                key=lambda s: (s.rerank_score is not None, s.rerank_score or 0.0),
                reverse=True,
            )
            # Past the row cap, pin a fixed height so the table scrolls instead of
            # growing the page; below it, leave height auto so short tables stay compact.
            df_kwargs = {}
            if len(grounded) > _SOURCES_TABLE_MAX_ROWS:
                df_kwargs["height"] = (_SOURCES_TABLE_MAX_ROWS + 1) * _DATAFRAME_ROW_PX + 3
            st.dataframe(
                [
                    {
                        "Source": _source_title(s.document),
                        "ReRank": _score_str(s.rerank_score),
                        "Similarity": _score_str(s.cosine_score),
                        "Chunk text": s.document.content,
                    }
                    for s in grounded
                ],
                width="stretch",
                hide_index=True,
                **df_kwargs,
            )


if __name__ == "__main__":
    main()
