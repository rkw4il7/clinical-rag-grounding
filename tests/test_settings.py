"""Tests for the settings loader (spec §6, §4.5)."""

from __future__ import annotations

import pytest

from corpus_rag.settings import Settings, SourceConfig


def _settings() -> Settings:
    # _env_file=None: ignore any developer .env so defaults are deterministic.
    return Settings(_env_file=None)


def test_defaults(clean_env) -> None:
    s = _settings()
    assert s.embed_model_id == "BAAI/bge-base-en-v1.5"
    assert s.embedding_dim is None
    assert s.top_k == 10
    assert s.min_score == 0.35  # hard grounding gate ON by default (fail-safe)
    assert s.ocr_on is True  # OCR on by default (clinical scans)
    assert s.llm_model == "local-model"
    assert s.corpus_sources == []
    assert s.pg_conn_str.startswith("postgresql://")


def test_env_overrides(clean_env) -> None:
    clean_env.setenv("EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
    clean_env.setenv("EMBEDDING_DIM", "384")
    clean_env.setenv("TOP_K", "5")
    clean_env.setenv("MIN_SCORE", "0.25")
    clean_env.setenv("LLM_BASE_URL", "http://llm:9000/v1")
    clean_env.setenv("LLM_MODEL", "qwen")
    clean_env.setenv("PG_CONN_STR", "postgresql://u:p@db:5432/x")

    s = _settings()
    assert s.embed_model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.embedding_dim == 384
    assert s.top_k == 5
    assert s.min_score == 0.25
    assert s.llm_base_url == "http://llm:9000/v1"
    assert s.llm_model == "qwen"
    assert s.pg_conn_str == "postgresql://u:p@db:5432/x"


def test_case_insensitive_env(clean_env) -> None:
    clean_env.setenv("top_k", "7")
    assert _settings().top_k == 7


def test_corpus_sources_json_decode(clean_env) -> None:
    clean_env.setenv(
        "CORPUS_SOURCES",
        '[{"adapter": "local_path", "root": "tests/data/**/*"},'
        ' {"adapter": "url", "url": "https://example.com/a.pdf"}]',
    )
    s = _settings()
    assert len(s.corpus_sources) == 2
    assert all(isinstance(src, SourceConfig) for src in s.corpus_sources)
    assert s.corpus_sources[0].adapter == "local_path"
    assert s.corpus_sources[0].root == "tests/data/**/*"
    assert s.corpus_sources[1].adapter == "url"
    assert s.corpus_sources[1].url == "https://example.com/a.pdf"


def test_corpus_sources_extra_fields_allowed(clean_env) -> None:
    clean_env.setenv(
        "CORPUS_SOURCES",
        '[{"adapter": "local_path", "root": "x", "recursive": true}]',
    )
    s = _settings()
    assert s.corpus_sources[0].adapter == "local_path"
    # extra="allow" keeps unknown keys on the model.
    assert s.corpus_sources[0].model_dump()["recursive"] is True


def test_invalid_top_k_rejected(clean_env) -> None:
    clean_env.setenv("TOP_K", "0")
    with pytest.raises(ValueError):
        _settings()


def test_invalid_embedding_dim_rejected(clean_env) -> None:
    clean_env.setenv("EMBEDDING_DIM", "0")
    with pytest.raises(ValueError):
        _settings()


@pytest.mark.parametrize("bad", ["-0.5", "1.5", "2.0"])
def test_invalid_min_score_rejected(clean_env, bad: str) -> None:
    clean_env.setenv("MIN_SCORE", bad)
    with pytest.raises(ValueError, match="MIN_SCORE"):
        _settings()


def test_rerank_defaults(clean_env) -> None:
    s = _settings()
    assert s.rerank_candidates == 20
    assert s.rerank_model_id == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert s.llm_timeout == 180


def test_rerank_and_timeout_env_overrides(clean_env) -> None:
    clean_env.setenv("RERANK_CANDIDATES", "30")
    clean_env.setenv("RERANK_MODEL_ID", "BAAI/bge-reranker-base")
    clean_env.setenv("LLM_TIMEOUT", "90")

    s = _settings()
    assert s.rerank_candidates == 30
    assert s.rerank_model_id == "BAAI/bge-reranker-base"
    assert s.llm_timeout == 90


def test_invalid_rerank_candidates_rejected(clean_env) -> None:
    clean_env.setenv("RERANK_CANDIDATES", "0")
    with pytest.raises(ValueError, match="RERANK_CANDIDATES"):
        _settings()


def test_invalid_llm_timeout_rejected(clean_env) -> None:
    clean_env.setenv("LLM_TIMEOUT", "0")
    with pytest.raises(ValueError, match="LLM_TIMEOUT"):
        _settings()
