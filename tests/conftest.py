"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

# Env vars that Settings reads; cleared per-test for deterministic parsing.
_SETTINGS_ENV_VARS = (
    "PG_CONN_STR",
    "EMBED_MODEL_ID",
    "EMBEDDING_DIM",
    "TOP_K",
    "MIN_SCORE",
    "CHUNK_TOKEN_MARGIN",
    "ENABLE_OCR",
    "RERANK_CANDIDATES",
    "RERANK_MODEL_ID",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_TIMEOUT",
    "CORPUS_SOURCES",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove all Settings env vars for deterministic parsing.

    Yields monkeypatch so tests can set individual vars on a known baseline.
    Tests construct ``Settings(_env_file=None)`` to also ignore any local
    ``.env`` file.
    """
    for var in _SETTINGS_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch
