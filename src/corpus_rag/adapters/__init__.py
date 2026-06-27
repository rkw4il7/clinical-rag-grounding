"""Source-adapter registry.

Maps an ``adapter`` key (from each ``CORPUS_SOURCES`` entry) to a concrete
adapter and resolves every configured origin into a flat list of ingestible
sources. New origins = new adapter + one registry entry (root ``spec.md`` §3.1).
"""

from __future__ import annotations

from collections.abc import Callable

from corpus_rag.adapters.base import Source, SourceAdapter
from corpus_rag.adapters.local_path import LocalPathAdapter
from corpus_rag.adapters.url import UrlAdapter
from corpus_rag.settings import SourceConfig

# Builders take a SourceConfig and return a ready SourceAdapter. Each is
# responsible for validating the fields it needs.
_REGISTRY: dict[str, Callable[[SourceConfig], SourceAdapter]] = {
    "local_path": lambda cfg: LocalPathAdapter(root=cfg.root or ""),
    "url": lambda cfg: UrlAdapter(url=cfg.url or ""),
}


def build_adapter(config: SourceConfig) -> SourceAdapter:
    """Resolve one ``SourceConfig`` to a concrete adapter.

    :raises ValueError: If ``config.adapter`` is not a registered key.
    """
    builder = _REGISTRY.get(config.adapter)
    if builder is None:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown source adapter {config.adapter!r}; known adapters: {known}")
    return builder(config)


def discover_all(configs: list[SourceConfig]) -> list[Source]:
    """Resolve and run every configured adapter; return all sources flattened."""
    sources: list[Source] = []
    for cfg in configs:
        sources.extend(build_adapter(cfg).discover())
    return sources


__all__ = [
    "LocalPathAdapter",
    "Source",
    "SourceAdapter",
    "UrlAdapter",
    "build_adapter",
    "discover_all",
]
