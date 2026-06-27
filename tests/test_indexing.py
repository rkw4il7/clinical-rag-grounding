"""Tests for the indexing pipeline.

Offline tests cover the source-input mapping with a mock pipeline (no models,
no DB). Live tests (``@pytest.mark.live``) build the real pipeline and ingest a
sample corpus; they need the embedding model + Postgres/pgvector and are skipped
offline. The §7.3/§7.4 ingest checks additionally require a sample corpus under
``tests/data`` (pending F4 confirmation) and self-skip if it is absent.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from corpus_rag.pipelines.indexing import run_indexing

_DATA_DIR = Path(__file__).parent / "data"


# --- offline -------------------------------------------------------------


def test_run_indexing_maps_sources_to_converter() -> None:
    pipeline = MagicMock()
    sources = ["a.pdf", "b.html"]

    run_indexing(pipeline, sources)

    pipeline.run.assert_called_once_with({"converter": {"sources": sources}})


def test_run_indexing_returns_pipeline_result() -> None:
    pipeline = MagicMock()
    pipeline.run.return_value = {"writer": {"documents_written": 7}}

    result = run_indexing(pipeline, ["a.pdf"])

    assert result == {"writer": {"documents_written": 7}}


# --- live ----------------------------------------------------------------


@pytest.mark.live
def test_build_indexing_pipeline_wires_components() -> None:
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    from corpus_rag.pipelines.indexing import build_indexing_pipeline

    pipeline = build_indexing_pipeline(InMemoryDocumentStore())
    graph = pipeline.to_dict()

    assert set(graph["components"]) == {"converter", "embedder", "writer"}
    edges = {(c["sender"].split(".")[0], c["receiver"].split(".")[0]) for c in graph["connections"]}
    assert ("converter", "embedder") in edges
    assert ("embedder", "writer") in edges


# Corpus formats in scope this phase (spec §7.3: mixed PDF+DOCX+HTML). Other
# Docling-supported formats are deferred — keep the allowlist to spec scope so a
# stray .csv/.xlsx doesn't silently produce empty chunks.
_SAMPLE_EXTS = {".pdf", ".docx", ".html", ".htm"}


def _sample_sources() -> list[str]:
    if not _DATA_DIR.is_dir():
        return []
    # Allowlist known formats so stray files (.gitkeep, .txt, …) never reach Docling.
    return sorted(
        str(p) for p in _DATA_DIR.rglob("*") if p.is_file() and p.suffix.lower() in _SAMPLE_EXTS
    )


@pytest.mark.live
def test_ingest_writes_embedded_chunks_with_provenance() -> None:
    """§7.3: mixed corpus writes N>0 chunks, each embedded + with source meta."""
    sources = _sample_sources()
    if not sources:
        pytest.skip("No sample corpus in tests/data (pending F4 confirmation).")

    from corpus_rag.document_store import build_document_store
    from corpus_rag.embeddings import resolve_embedding_dim
    from corpus_rag.pipelines.indexing import build_indexing_pipeline
    from corpus_rag.settings import get_settings

    settings = get_settings()
    dim = resolve_embedding_dim(settings.embed_model_id)
    store = build_document_store(settings, recreate_table=True)
    run_indexing(build_indexing_pipeline(store, settings), sources)

    docs = store.filter_documents()
    assert len(docs) > 0
    for doc in docs:
        assert doc.embedding is not None and len(doc.embedding) == dim
        assert doc.meta  # non-empty provenance


@pytest.mark.live
def test_reingest_is_idempotent() -> None:
    """§7.4: re-ingesting the same docs does not grow the chunk count."""
    sources = _sample_sources()
    if not sources:
        pytest.skip("No sample corpus in tests/data (pending F4 confirmation).")

    from corpus_rag.document_store import build_document_store
    from corpus_rag.pipelines.indexing import build_indexing_pipeline
    from corpus_rag.settings import get_settings

    settings = get_settings()
    store = build_document_store(settings, recreate_table=True)
    pipeline = build_indexing_pipeline(store, settings)

    run_indexing(pipeline, sources)
    first = store.count_documents()
    run_indexing(pipeline, sources)
    second = store.count_documents()

    assert second == first


@pytest.mark.live
def test_chunks_respect_embedding_token_budget() -> None:
    """No emitted chunk exceeds the embedding model's token budget.

    Guards against silent embed-time truncation: the embedder would otherwise
    encode only the start of an over-long chunk while the full text is still
    stored/displayed. The chunker is capped to ``max_tokens - margin``.
    """
    sources = _sample_sources()
    if not sources:
        pytest.skip("No sample corpus in tests/data (pending F4 confirmation).")

    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from haystack_integrations.components.converters.docling import (
        DoclingConverter,
        ExportType,
    )

    from corpus_rag.pipelines.indexing import build_chunker, build_converter, chunk_token_budget
    from corpus_rag.settings import get_settings

    settings = get_settings()
    budget = chunk_token_budget(settings.embed_model_id, settings.chunk_token_margin)
    converter = DoclingConverter(
        converter=build_converter(settings),
        export_type=ExportType.DOC_CHUNKS,
        chunker=build_chunker(settings.embed_model_id, token_margin=settings.chunk_token_margin),
    )
    docs = converter.run(sources=sources)["documents"]
    assert docs, "no chunks emitted from the sample corpus"

    tokenizer = HuggingFaceTokenizer.from_pretrained(model_name=settings.embed_model_id)
    counts = [tokenizer.count_tokens(d.content) for d in docs]
    oversized = [c for c in counts if c > budget]
    assert not oversized, f"chunks exceed token budget {budget}: {oversized}"
