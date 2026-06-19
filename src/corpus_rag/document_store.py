"""Vector store construction + embedding-dimension contract.

Builds the ``PgvectorDocumentStore`` from settings (root ``spec.md`` §4):
cosine similarity, HNSW ANN index, and ``recreate_table=False`` in normal
runs (``True`` only behind an explicit dev reset).

Enforces the §7.2 hard contract: the embedding model's true output dimension
must equal the store's ``embedding_dimension``. Mismatch fails fast at
construction time, before any documents are written or queried.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from haystack.utils import Secret
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

from corpus_rag.embeddings import resolve_embedding_dim
from corpus_rag.settings import Settings, get_settings

if TYPE_CHECKING:
    from haystack.document_stores.types import DocumentStore


class EmbeddingDimensionError(RuntimeError):
    """Raised when the embedding dimension contract (§7.2) is violated."""


def assert_dimension_contract(model_dim: int, store: DocumentStore) -> None:
    """Assert the embedder/store dimension contract (root ``spec.md`` §7.2).

    :param model_dim: The embedding model's true output dimension.
    :param store: A document store exposing ``embedding_dimension``.
    :raises EmbeddingDimensionError: If the dimensions differ.
    """
    store_dim = getattr(store, "embedding_dimension", None)
    if store_dim != model_dim:
        raise EmbeddingDimensionError(
            "Embedding dimension contract violated: model emits "
            f"{model_dim} dims but document store expects {store_dim}. "
            "The text/document embedder model must match the store."
        )


def build_document_store(
    settings: Settings | None = None,
    *,
    recreate_table: bool = False,
) -> PgvectorDocumentStore:
    """Construct the pgvector document store and enforce the dim contract.

    Derives the embedding dimension from ``EMBED_MODEL_ID`` (the authoritative
    source), constructs the store with cosine similarity + HNSW, and asserts
    the §7.2 contract. If ``EMBEDDING_DIM`` is set in settings as a double-check,
    a disagreement with the resolved model dimension also fails fast.

    :param settings: Settings to read from; defaults to the cached process settings.
    :param recreate_table: Drop + recreate the table (dev reset only). Defaults
        to ``False`` per root ``spec.md`` §4.
    :returns: A configured ``PgvectorDocumentStore``.
    :raises EmbeddingDimensionError: If a configured ``EMBEDDING_DIM`` disagrees
        with the model, or the constructed store's dimension disagrees.
    """
    settings = settings or get_settings()

    model_dim = resolve_embedding_dim(settings.embed_model_id)

    # Optional fast-fail double-check: if the operator pinned EMBEDDING_DIM, it
    # must agree with the model's real dimension before we touch the database.
    if settings.embedding_dim is not None and settings.embedding_dim != model_dim:
        raise EmbeddingDimensionError(
            f"Configured EMBEDDING_DIM={settings.embedding_dim} does not match "
            f"model {settings.embed_model_id!r} which emits {model_dim} dims."
        )

    store = PgvectorDocumentStore(
        connection_string=Secret.from_token(settings.pg_conn_str),
        embedding_dimension=model_dim,
        vector_function="cosine_similarity",
        search_strategy="hnsw",
        recreate_table=recreate_table,
    )

    assert_dimension_contract(model_dim, store)
    return store
