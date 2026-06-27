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
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
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


def build_converter(settings: Settings | None = None) -> DocumentConverter:
    """Docling ``DocumentConverter`` with OCR gated by ``OCR_ON``.

    OCR is on by default — a clinical corpus commonly contains scanned/faxed
    pages, and missing their text silently is worse than the extra ingest time.
    Set ``OCR_ON=false`` for born-digital-only corpora to skip image-region OCR
    (text layers still extract either way). Only the PDF pipeline carries the flag.
    """
    settings = settings or get_settings()
    pdf_options = PdfPipelineOptions(do_ocr=settings.ocr_on)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)}
    )


def embedding_max_tokens(model_id: str) -> int:
    """The embedding model's max input tokens (from its tokenizer config)."""
    return HuggingFaceTokenizer.from_pretrained(model_name=model_id).get_max_tokens()


def chunk_token_budget(model_id: str, token_margin: int) -> int:
    """Hard per-chunk token budget: embedding max tokens minus the safety margin."""
    return max(1, embedding_max_tokens(model_id) - token_margin)


def build_chunker(model_id: str, *, token_margin: int = 16) -> HybridChunker:
    """Build a HybridChunker capped to the embedding model's token budget.

    The chunker's tokenizer ``max_tokens`` is set to ``embedding_max_tokens -
    token_margin`` so no emitted chunk exceeds what the embedder can encode — the
    embedder would otherwise silently truncate the tail, making the embedding
    represent only the start of a chunk whose full text is still displayed/stored.
    The margin leaves room for the embedder's special tokens plus headroom.

    Passing an explicit ``HuggingFaceTokenizer`` (not the bare model-id string)
    also avoids docling's deprecation of the string form.
    """
    budget = chunk_token_budget(model_id, token_margin)
    tokenizer = HuggingFaceTokenizer.from_pretrained(model_name=model_id, max_tokens=budget)
    return HybridChunker(tokenizer=tokenizer)


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
        converter=build_converter(settings),
        export_type=ExportType.DOC_CHUNKS,
        chunker=build_chunker(settings.embed_model_id, token_margin=settings.chunk_token_margin),
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
    # Explicit socket names: the embedder exposes both `documents` and `meta`
    # outputs, so auto-connect to the writer is ambiguous in Haystack 2.x.
    pipeline.connect("converter.documents", "embedder.documents")
    pipeline.connect("embedder.documents", "writer.documents")
    return pipeline


def run_indexing(pipeline: Pipeline, sources: list[Source]) -> dict:
    """Run the indexing pipeline over already-discovered sources.

    :param pipeline: A pipeline from :func:`build_indexing_pipeline`.
    :param sources: Paths / ByteStreams from the adapter registry.
    :returns: The pipeline result (includes the writer's documents-written count).
    """
    return pipeline.run({"converter": {"sources": sources}})
