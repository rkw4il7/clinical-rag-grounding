"""Typed settings loader for the Corpus RAG Explorer.

Loads every environment variable named in root ``spec.md`` §6 (plus the
``EMBEDDING_DIM`` / ``MIN_SCORE`` knobs introduced in the task spec) into a
single validated ``Settings`` object. Complex fields (``CORPUS_SOURCES``) are
JSON-decoded from their env value by pydantic-settings.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Streamlit's per-file ceiling shipped in .streamlit/config.toml. UPLOAD_MAX_MB
# must stay <= this or Streamlit rejects the upload at its transport layer before
# the app's own check runs. Kept in sync with that file by convention.
_STREAMLIT_MAX_UPLOAD_MB = 200


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
    # Abstention floor (cosine, 0..1). Default non-zero so the §2A grounding gate
    # is ON by default: below this, chunks are dropped and the app abstains BEFORE
    # generating. Setting 0.0 disables the hard gate (grounding then rests only on
    # the prompt) — fail-open, so build_query_engine/build_rerank_engine warn.
    min_score: float = 0.35

    # Chunking: cap each chunk to the embedding model's max tokens MINUS this
    # margin, so no chunk is silently truncated at embed time (the margin leaves
    # room for the embedder's special tokens + a little headroom).
    chunk_token_margin: int = 16

    # OCR during ingest (env: OCR_ON). On by default — a clinical corpus often
    # includes scanned/faxed pages, and missing their text silently is worse than
    # the extra ingest time. Set OCR_ON=false for born-digital-only corpora to
    # skip image-region OCR (text layers still extract either way).
    ocr_on: bool = True

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

    # GUI upload cap (MB) for the total of a single upload batch. Ingest runs in
    # the Streamlit request handler, so this bounds how long a worker can block.
    # Must stay <= Streamlit's server.maxUploadSize (.streamlit/config.toml).
    upload_max_mb: int = 200

    @model_validator(mode="after")
    def _validate(self) -> Settings:
        if self.top_k < 1:
            raise ValueError("TOP_K must be >= 1")
        if self.embedding_dim is not None and self.embedding_dim < 1:
            raise ValueError("EMBEDDING_DIM, when set, must be >= 1")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("MIN_SCORE must be in [0.0, 1.0] (cosine floor)")
        if self.chunk_token_margin < 0:
            raise ValueError("CHUNK_TOKEN_MARGIN must be >= 0")
        if self.upload_max_mb < 1:
            raise ValueError("UPLOAD_MAX_MB must be >= 1")
        if self.upload_max_mb > _STREAMLIT_MAX_UPLOAD_MB:
            warnings.warn(
                f"UPLOAD_MAX_MB={self.upload_max_mb} exceeds Streamlit's "
                f"maxUploadSize ({_STREAMLIT_MAX_UPLOAD_MB} MB in "
                ".streamlit/config.toml). Raise [server] maxUploadSize too, or "
                "Streamlit rejects the upload before the app's check runs.",
                stacklevel=2,
            )
        if self.rerank_candidates < 1:
            raise ValueError("RERANK_CANDIDATES must be >= 1")
        if self.llm_timeout < 1:
            raise ValueError("LLM_TIMEOUT must be >= 1 (seconds)")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return process-wide cached settings."""
    return Settings()
