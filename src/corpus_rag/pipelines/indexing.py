"""Indexing pipeline: sources -> chunks -> embeddings -> vector store.

Implements root ``spec.md`` §3.2:

    DoclingConverter(DOC_CHUNKS, HybridChunker(tokenizer=EMBED_MODEL_ID))
      -> SentenceTransformersDocumentEmbedder(model=EMBED_MODEL_ID)
      -> DocumentWriter(policy=OVERWRITE)

The chunker tokenizer and the document embedder use the SAME ``EMBED_MODEL_ID``
so chunk boundaries align with the embedding model and re-ingest stays
idempotent (OVERWRITE on the content-derived id).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from docling.chunking import HybridChunker
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.converters.docling import (
    DoclingConverter,
    ExportType,
)

from corpus_rag.adapters.base import Source
from corpus_rag.settings import Settings, get_settings

if TYPE_CHECKING:
    from haystack.document_stores.types import DocumentStore


def build_indexing_pipeline(
    document_store: DocumentStore,
    settings: Settings | None = None,
) -> Pipeline:
    """Construct the Docling -> embedder -> writer indexing pipeline.

    :param document_store: Target store (its ``embedding_dimension`` must match
        the embedder; enforce via ``build_document_store`` beforehand).
    :param settings: Settings to read ``EMBED_MODEL_ID`` from; defaults to cached.
    :returns: A wired (un-warmed) Haystack ``Pipeline``.
    """
    settings = settings or get_settings()

    converter = DoclingConverter(
        export_type=ExportType.DOC_CHUNKS,
        chunker=HybridChunker(tokenizer=settings.embed_model_id),
    )
    embedder = SentenceTransformersDocumentEmbedder(model=settings.embed_model_id)
    writer = DocumentWriter(
        document_store=document_store,
        policy=DuplicatePolicy.OVERWRITE,
    )

    pipeline = Pipeline()
    pipeline.add_component("converter", converter)
    pipeline.add_component("embedder", embedder)
    pipeline.add_component("writer", writer)
    pipeline.connect("converter", "embedder")
    pipeline.connect("embedder", "writer")
    return pipeline


def run_indexing(pipeline: Pipeline, sources: list[Source]) -> dict:
    """Run the indexing pipeline over already-discovered sources.

    :param pipeline: A pipeline from :func:`build_indexing_pipeline`.
    :param sources: Paths / ByteStreams from the adapter registry.
    :returns: The pipeline result (includes the writer's documents-written count).
    """
    return pipeline.run({"converter": {"sources": sources}})
