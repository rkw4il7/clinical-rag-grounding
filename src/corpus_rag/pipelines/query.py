"""Query path: embed -> retrieve -> GATE -> generate (root ``spec.md`` §3.3, §2A).

    SentenceTransformersTextEmbedder(EMBED_MODEL_ID)
      -> PgvectorEmbeddingRetriever(top_k=TOP_K)
      -> [§2A grounding gate: MIN_SCORE floor; abstain if nothing qualifies]
      -> PromptBuilder(RAG_PROMPT_TEMPLATE)
      -> OpenAIGenerator(local base_url, temperature=0)

``run_query`` drives the components STEP BY STEP and applies the grounding gate
BEFORE invoking the generator: the LLM is never called until retrieval has
established grounding (no chunk at/above ``MIN_SCORE`` → abstain, generator not
invoked). This is the healthcare-safety ordering — generation never precedes
grounding. It returns ``(answer, documents)`` with documents in cosine-ranked
order, always surfaced (even on abstention) so the UI shows what grounding existed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING

from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pgvector import (
    PgvectorEmbeddingRetriever,
)

from corpus_rag.prompts import ABSTENTION_ANSWER, RAG_PROMPT_TEMPLATE
from corpus_rag.settings import Settings, get_settings

if TYPE_CHECKING:
    from haystack import Document
    from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

logger = logging.getLogger(__name__)


def _warn_if_gate_open(settings: Settings) -> None:
    """Warn when MIN_SCORE == 0: the hard grounding gate is disabled (fail-open)."""
    if settings.min_score <= 0.0:
        logger.warning(
            "MIN_SCORE=%s disables the §2A hard grounding gate: the retriever "
            "always returns top_k and the generator always runs, so grounding "
            "rests on the prompt alone. Set MIN_SCORE > 0 for the mechanical gate.",
            settings.min_score,
        )


def _build_generator(settings: Settings) -> OpenAIGenerator:
    return OpenAIGenerator(
        # Local OpenAI-compatible servers ignore the key but the client requires
        # a non-empty value; never read a real OPENAI_API_KEY from the env.
        api_key=Secret.from_token("not-needed-for-local-server"),
        model=settings.llm_model,
        api_base_url=settings.llm_base_url,
        generation_kwargs={"temperature": 0},  # §7.7 reproducible answers
        timeout=settings.llm_timeout,
    )


@dataclass
class QueryEngine:
    """Warmed components for the gate-before-generate query path (built once)."""

    text_embedder: SentenceTransformersTextEmbedder
    retriever: PgvectorEmbeddingRetriever
    prompt_builder: PromptBuilder
    generator: OpenAIGenerator


def build_query_engine(
    document_store: PgvectorDocumentStore,
    settings: Settings | None = None,
) -> QueryEngine:
    """Build + warm the embed → retrieve → gate → generate components.

    Components are driven directly (not via a Haystack ``Pipeline``) so the
    §2A grounding gate can run between retrieval and generation — the generator
    is only invoked once grounding is established.
    """
    settings = settings or get_settings()
    _warn_if_gate_open(settings)

    text_embedder = SentenceTransformersTextEmbedder(model=settings.embed_model_id)
    text_embedder.warm_up()
    retriever = PgvectorEmbeddingRetriever(
        document_store=document_store,
        top_k=settings.top_k,
    )
    prompt_builder = PromptBuilder(
        template=RAG_PROMPT_TEMPLATE,
        required_variables=["query", "documents"],
    )
    return QueryEngine(text_embedder, retriever, prompt_builder, _build_generator(settings))


@lru_cache(maxsize=1)
def _default_query_engine() -> QueryEngine:
    """Process-wide query engine (built once; loads the embedder + store)."""
    from corpus_rag.document_store import build_document_store

    settings = get_settings()
    return build_query_engine(build_document_store(settings), settings)


def run_query(
    query: str,
    *,
    engine: QueryEngine | None = None,
    settings: Settings | None = None,
) -> tuple[str, list[Document]]:
    """Run a grounded query, gating BEFORE generation (root §4.3, §2A).

    Order: embed → retrieve → apply the ``MIN_SCORE`` grounding gate → abstain if
    nothing qualifies (the generator is NOT called) → otherwise build the prompt
    and generate. This guarantees the LLM never runs without established grounding.

    :param query: The user's question.
    :param engine: Override engine (defaults to the cached process engine).
    :param settings: Override settings (defaults to cached process settings).
    :returns: ``(answer, documents)`` — documents in cosine-ranked order, always
        returned even on abstention.
    """
    settings = settings or get_settings()
    engine = engine or _default_query_engine()

    # Empty/whitespace query → undefined embedding behavior; abstain immediately.
    if not query.strip():
        return ABSTENTION_ANSWER, []

    embedding = engine.text_embedder.run(text=query)["embedding"]
    documents: list[Document] = engine.retriever.run(query_embedding=embedding)["documents"]

    # Grounding gate (§2A.3): drop chunks below the MIN_SCORE floor; abstain when
    # nothing remains — BEFORE the generator runs, so no ungrounded prose is ever
    # produced (not merely discarded).
    if settings.min_score > 0.0:
        documents = [d for d in documents if (d.score or 0.0) >= settings.min_score]

    if not documents:
        return ABSTENTION_ANSWER, documents

    prompt = engine.prompt_builder.run(query=query, documents=documents)["prompt"]
    replies = engine.generator.run(prompt=prompt).get("replies") or []
    if not replies:
        # Distinct from a grounding abstention: the LLM produced nothing (network
        # error, context overflow, refusal). Same return, but make it detectable.
        logger.warning("Generator returned no replies for query %r", query)
        return ABSTENTION_ANSWER, documents
    return replies[0], documents


# --- reranking (post-retrieval) -----------------------------------------
#
# A cross-encoder reranks the top cosine candidates. We capture each chunk's
# cosine rank+score BEFORE reranking (the ranker mutates Document.score in place,
# aliasing the retriever's objects), then the rerank rank+score AFTER, so the UI
# can show both and prove the rerank reorders the initial cosine list.


@dataclass(frozen=True)
class RankedSource:
    """One retrieved chunk with both its cosine and rerank rank/score.

    ``used_for_grounding`` is True only for the chunks actually fed to the
    generator (the top-TOP_K of the floor-passing set) — so the UI can label
    exactly what grounded the answer, not merely what passed the floor.
    """

    document: Document
    cosine_rank: int
    cosine_score: float | None
    rerank_rank: int
    rerank_score: float | None
    used_for_grounding: bool = False


@dataclass
class RerankEngine:
    """Warmed components for a rerank query (built once; heavy models)."""

    text_embedder: SentenceTransformersTextEmbedder
    retriever: PgvectorEmbeddingRetriever
    ranker: SentenceTransformersSimilarityRanker
    prompt_builder: PromptBuilder
    generator: OpenAIGenerator


def build_rerank_engine(
    document_store: PgvectorDocumentStore,
    settings: Settings | None = None,
) -> RerankEngine:
    """Build + warm up the components for cosine-retrieve → cross-encoder rerank."""
    settings = settings or get_settings()
    _warn_if_gate_open(settings)

    text_embedder = SentenceTransformersTextEmbedder(model=settings.embed_model_id)
    text_embedder.warm_up()
    retriever = PgvectorEmbeddingRetriever(
        document_store=document_store,
        top_k=settings.rerank_candidates,
    )
    ranker = SentenceTransformersSimilarityRanker(
        model=settings.rerank_model_id,
        top_k=settings.rerank_candidates,
    )
    ranker.warm_up()
    prompt_builder = PromptBuilder(
        template=RAG_PROMPT_TEMPLATE,
        required_variables=["query", "documents"],
    )
    return RerankEngine(
        text_embedder, retriever, ranker, prompt_builder, _build_generator(settings)
    )


@lru_cache(maxsize=1)
def _default_rerank_engine() -> RerankEngine:
    from corpus_rag.document_store import build_document_store

    settings = get_settings()
    return build_rerank_engine(build_document_store(settings), settings)


def run_query_reranked(
    query: str,
    *,
    engine: RerankEngine | None = None,
    settings: Settings | None = None,
) -> tuple[str, list[RankedSource]]:
    """Run a grounded query with cross-encoder reranking.

    Retrieves ``RERANK_CANDIDATES`` chunks by cosine similarity, reranks them
    with a cross-encoder, and grounds the answer in the reranked order.

    :returns: ``(answer, ranked_sources)`` — ``ranked_sources`` in RERANK order,
        each carrying both the cosine rank/score and the rerank rank/score.
        Always returned (even on abstention) so the UI can show the comparison.
    """
    settings = settings or get_settings()
    engine = engine or _default_rerank_engine()

    if not query.strip():
        return ABSTENTION_ANSWER, []

    embedding = engine.text_embedder.run(text=query)["embedding"]
    cosine_docs = engine.retriever.run(query_embedding=embedding)["documents"]

    # Snapshot cosine rank+score NOW (the ranker overwrites Document.score). Key
    # by Document.id, NOT Python object identity: the ranker returns NEW Document
    # objects (copies), so id() would not match the cosine snapshot. Document.id
    # is a content+meta hash; the pgvector store keys rows by id (OVERWRITE), so
    # the retrieved set has unique ids — no collision to worry about here.
    cosine_by_id = {doc.id: (rank, doc.score) for rank, doc in enumerate(cosine_docs, start=1)}

    reranked_docs = engine.ranker.run(query=query, documents=cosine_docs)["documents"]

    ranked_sources = [
        RankedSource(
            document=doc,
            cosine_rank=cosine_by_id[doc.id][0],
            cosine_score=cosine_by_id[doc.id][1],
            rerank_rank=rerank_rank,
            rerank_score=doc.score,
        )
        for rerank_rank, doc in enumerate(reranked_docs, start=1)
    ]

    # Grounding gate (§2A.3) stays on the COSINE score (the MIN_SCORE floor is a
    # cosine threshold), independent of the rerank reordering.
    grounded = [
        rs
        for rs in ranked_sources
        if settings.min_score <= 0.0 or (rs.cosine_score or 0.0) >= settings.min_score
    ]
    if not grounded:
        return ABSTENTION_ANSWER, ranked_sources

    # Ground the answer in the TOP_K reranked chunks only — the whole point of
    # reranking is to feed the LLM the best few, not all RERANK_CANDIDATES.
    llm_sources = grounded[: settings.top_k]

    # Mark exactly the chunks fed to the generator so the UI labels truth (not
    # "every floor-passer"). llm_sources are members of ranked_sources here, so
    # identity membership is precise; replace() rebuilds the frozen dataclasses.
    used_ids = {id(rs) for rs in llm_sources}
    ranked_sources = [replace(rs, used_for_grounding=id(rs) in used_ids) for rs in ranked_sources]

    prompt = engine.prompt_builder.run(query=query, documents=[rs.document for rs in llm_sources])[
        "prompt"
    ]
    replies = engine.generator.run(prompt=prompt).get("replies") or []
    if not replies:
        logger.warning("Generator returned no replies for query %r", query)
        return ABSTENTION_ANSWER, ranked_sources
    return replies[0], ranked_sources
