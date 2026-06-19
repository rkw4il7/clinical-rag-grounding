"""Tests for the document-store builder and §7.2 dimension contract.

Fully offline: the sentence-transformers model load and the real
``PgvectorDocumentStore`` (which would need torch / a live database) are
patched out. We assert the *wiring* and the *contract*, not the libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from corpus_rag import document_store as ds
from corpus_rag.document_store import (
    EmbeddingDimensionError,
    assert_dimension_contract,
    build_document_store,
)
from corpus_rag.settings import Settings


@dataclass
class _FakeStore:
    """Stand-in for PgvectorDocumentStore capturing construction kwargs."""

    kwargs: dict[str, Any]

    @property
    def embedding_dimension(self) -> int:
        return self.kwargs["embedding_dimension"]


@pytest.fixture
def fake_store_factory(monkeypatch: pytest.MonkeyPatch):
    """Patch PgvectorDocumentStore with a capturing fake; return the holder."""
    captured: dict[str, Any] = {}

    def _factory(**kwargs: Any) -> _FakeStore:
        captured.update(kwargs)
        return _FakeStore(kwargs=kwargs)

    monkeypatch.setattr(ds, "PgvectorDocumentStore", _factory)
    return captured


def _settings(**overrides: Any) -> Settings:
    base = dict(
        pg_conn_str="postgresql://u:p@localhost:5432/db",
        embed_model_id="fake/model",
        embedding_dim=None,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


# --- assert_dimension_contract -------------------------------------------


def test_contract_passes_on_match() -> None:
    store = _FakeStore(kwargs={"embedding_dimension": 768})
    # Should not raise.
    assert_dimension_contract(768, store)


def test_contract_raises_on_mismatch() -> None:
    store = _FakeStore(kwargs={"embedding_dimension": 384})
    with pytest.raises(EmbeddingDimensionError, match="contract violated"):
        assert_dimension_contract(768, store)


# --- build_document_store ------------------------------------------------


def test_build_uses_resolved_dim_and_locked_params(
    monkeypatch: pytest.MonkeyPatch, fake_store_factory: dict[str, Any]
) -> None:
    monkeypatch.setattr(ds, "resolve_embedding_dim", lambda _mid: 768)

    store = build_document_store(_settings())

    assert store.embedding_dimension == 768
    assert fake_store_factory["embedding_dimension"] == 768
    assert fake_store_factory["vector_function"] == "cosine_similarity"
    assert fake_store_factory["search_strategy"] == "hnsw"
    # Normal runs never recreate the table (root spec §4).
    assert fake_store_factory["recreate_table"] is False


def test_build_passes_recreate_table_flag(
    monkeypatch: pytest.MonkeyPatch, fake_store_factory: dict[str, Any]
) -> None:
    monkeypatch.setattr(ds, "resolve_embedding_dim", lambda _mid: 384)

    build_document_store(_settings(), recreate_table=True)

    assert fake_store_factory["recreate_table"] is True


def test_build_resolves_dim_from_configured_model(
    monkeypatch: pytest.MonkeyPatch, fake_store_factory: dict[str, Any]
) -> None:
    seen: list[str] = []

    def _resolve(model_id: str) -> int:
        seen.append(model_id)
        return 1024

    monkeypatch.setattr(ds, "resolve_embedding_dim", _resolve)

    build_document_store(_settings(embed_model_id="custom/embedder"))

    assert seen == ["custom/embedder"]
    assert fake_store_factory["embedding_dimension"] == 1024


def test_build_raises_when_configured_dim_disagrees(
    monkeypatch: pytest.MonkeyPatch, fake_store_factory: dict[str, Any]
) -> None:
    monkeypatch.setattr(ds, "resolve_embedding_dim", lambda _mid: 768)

    with pytest.raises(EmbeddingDimensionError, match="EMBEDDING_DIM=384"):
        build_document_store(_settings(embedding_dim=384))


def test_build_passes_when_configured_dim_agrees(
    monkeypatch: pytest.MonkeyPatch, fake_store_factory: dict[str, Any]
) -> None:
    monkeypatch.setattr(ds, "resolve_embedding_dim", lambda _mid: 768)

    store = build_document_store(_settings(embedding_dim=768))

    assert store.embedding_dimension == 768
