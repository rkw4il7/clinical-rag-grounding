"""URL source adapter.

Fetches a remote document and returns it as an in-memory ``ByteStream`` so the
converter never needs a local copy. The origin URL is stamped into the stream's
metadata so provenance survives into ``Document.meta`` (root ``spec.md`` §4).

Uses the stdlib ``urllib`` to avoid adding an HTTP dependency.
"""

from __future__ import annotations

import urllib.error
import urllib.request

from haystack.dataclasses import ByteStream

from corpus_rag.adapters.base import Source

_DEFAULT_TIMEOUT = 30


class UrlAdapter:
    """Fetch a single URL into a ``ByteStream``."""

    def __init__(self, url: str, *, timeout: int = _DEFAULT_TIMEOUT) -> None:
        if not url:
            raise ValueError("url adapter requires a non-empty 'url'")
        self.url = url
        self.timeout = timeout

    def discover(self) -> list[Source]:
        try:
            with urllib.request.urlopen(  # noqa: S310
                self.url, timeout=self.timeout
            ) as resp:
                data = resp.read()
                mime_type = resp.headers.get_content_type()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"url adapter failed to fetch {self.url!r}: {exc}") from exc
        stream = ByteStream(
            data=data,
            mime_type=mime_type,
            meta={"source_url": self.url},
        )
        return [stream]
