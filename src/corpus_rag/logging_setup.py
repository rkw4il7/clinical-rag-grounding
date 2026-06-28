"""Quiet known-noisy, non-actionable upstream log lines at interactive entrypoints.

These originate in transformers / sentence-transformers / Docling internals, not
this codebase, and only clutter interactive runs (app + CLI). Tests do NOT call
this, so the same signals still surface there.
"""

from __future__ import annotations

import logging
import warnings


def quiet_noisy_upstream() -> None:
    """Suppress two benign upstream messages seen during ingest/query.

    - ``tokenizer_kwargs ... renamed ... use processor_kwargs`` — a deprecation
      emitted inside sentence-transformers/transformers; we never pass that arg.
    - ``Token indices sequence length is longer than ... 512`` — the embedder and
      cross-encoder truncate over-long inputs internally; the chunk-token budget
      already bounds what we control, so this is informational, not an error.
    """
    # Scoped to transformers so an unrelated future warning mentioning the same
    # word elsewhere isn't silently dropped.
    warnings.filterwarnings("ignore", message=r".*tokenizer_kwargs.*", module=r"transformers.*")
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
