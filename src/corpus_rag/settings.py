"""Typed settings loader for the Corpus RAG Explorer.

Loads every environment variable named in root ``spec.md`` §6 (plus the
``EMBEDDING_DIM`` / ``MIN_SCORE`` knobs introduced in the task spec) into a
single validated ``Settings`` object. Complex fields (``CORPUS_SOURCES``) are
JSON-decoded from their env value by pydantic-settings.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SourceConfig(BaseModel):
    """One configured corpus origin (resolved by the adapter registry).

    ``adapter`` selects the registry entry (e.g. ``"local_path"`` or ``"url"``).
    ``root`` / ``url`` carry the origin location; which is required depends on
    the adapter and is validated by the adapter itself, not here.
    """

    model_config = ConfigDict(extra="allow")

    adapter: str
    root: str | None = None
    url: str | None = None


class Settings(BaseSettings):
    """Application settings sourced from env / ``.env`` (root spec §6)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Vector store connection (percent-encode special chars in the password).
    pg_conn_str: str = "postgresql://postgres:postgres@localhost:5432/corpus_rag"

    # Embedding model. EMBEDDING_DIM is derived at runtime from the model; the
    # optional override exists only as a fast-fail double-check (spec §1, §4.5).
    embed_model_id: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int | None = None

    # Retrieval / grounding knobs.
    top_k: int = 10
    min_score: float = 0.0  # abstention floor; 0.0 == off (spec §2A.3)

    # Reranking (post-retrieval). Retrieve RERANK_CANDIDATES by cosine, then a
    # cross-encoder reorders them; the UI shows both rankings side by side to
    # demonstrate the rerank overriding the initial cosine order.
    rerank_candidates: int = 20
    rerank_model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Local OpenAI-compatible generator endpoint.
    llm_base_url: str = "http://localhost:8080/v1"
    llm_model: str = "local-model"
    # Generous default: reasoning models emit long thinking traces and can take
    # well over the OpenAI client's 60s default before the first token.
    llm_timeout: int = 180

    # Corpus origins: JSON array of {adapter, root|url, ...} objects.
    corpus_sources: list[SourceConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> Settings:
        if self.top_k < 1:
            raise ValueError("TOP_K must be >= 1")
        if self.embedding_dim is not None and self.embedding_dim < 1:
            raise ValueError("EMBEDDING_DIM, when set, must be >= 1")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("MIN_SCORE must be in [0.0, 1.0] (cosine floor)")
        if self.rerank_candidates < 1:
            raise ValueError("RERANK_CANDIDATES must be >= 1")
        if self.llm_timeout < 1:
            raise ValueError("LLM_TIMEOUT must be >= 1 (seconds)")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return process-wide cached settings."""
    return Settings()
