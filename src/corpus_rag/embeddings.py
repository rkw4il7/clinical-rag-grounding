"""Embedding-model helpers.

The embedding dimension is a *hard contract* (root ``spec.md`` §4): the vector
store's ``embedding_dimension`` must equal the embedding model's true output
dimension. Rather than trust a hand-configured number, we derive it from the
loaded ``sentence-transformers`` model so the contract is anchored to reality.
"""

from __future__ import annotations

from functools import cache


@cache
def resolve_embedding_dim(model_id: str) -> int:
    """Return the output dimension of a sentence-transformers model.

    Loads the model (cached per ``model_id``) and queries its sentence
    embedding dimension. This is the authoritative value for the store's
    ``embedding_dimension`` and for the §7.2 dimension assertion.

    :param model_id: A sentence-transformers model id (``EMBED_MODEL_ID``).
    :returns: Positive embedding dimension.
    :raises RuntimeError: If the model reports no usable dimension.
    """
    # Imported lazily: loading sentence-transformers pulls in torch and is slow,
    # so callers that only need settings/store wiring don't pay for it.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    # sentence-transformers 5+ renamed get_sentence_embedding_dimension ->
    # get_embedding_dimension (the old name now emits a FutureWarning). Prefer the
    # new name, fall back for older installs.
    get_dim = (
        getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    )
    dim = get_dim()
    if not dim or dim < 1:
        raise RuntimeError(f"Embedding model {model_id!r} reported an invalid dimension: {dim!r}")
    return int(dim)
