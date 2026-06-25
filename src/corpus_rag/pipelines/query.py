"""Query pipeline: query -> retrieve -> grounded generation (root ``spec.md`` §3.3).

    SentenceTransformersTextEmbedder(EMBED_MODEL_ID)
      -> PgvectorEmbeddingRetriever(top_k=TOP_K)
      -> PromptBuilder(RAG_PROMPT_TEMPLATE)
      -> OpenAIGenerator(local base_url, temperature=0)

``run_query`` returns BOTH the generated answer and the retriever's documents in
cosine-ranked order (root §4.3). It enforces the §2A grounding contract: when the
retriever returns nothing (or nothing at/above ``MIN_SCORE``), it returns the
abstention answer and discards any generated prose — but still surfaces the
(possibly empty) document list so the UI always shows what grounding existed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.rankers import TransformersSimilarityRanker
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


def build_query_pipeline(
    document_store: PgvectorDocumentStore,
    settings: Settings | None = None,
) -> Pipeline:
    """Construct the text-embed -> retrieve -> prompt -> generate pipeline."""
    settings = settings or get_settings()

    text_embedder = SentenceTransformersTextEmbedder(model=settings.embed_model_id)
    retriever = PgvectorEmbeddingRetriever(
        document_store=document_store,
        top_k=settings.top_k,
    )
    prompt_builder = PromptBuilder(
        template=RAG_PROMPT_TEMPLATE,
        required_variables=["query", "documents"],
    )
    generator = OpenAIGenerator(
        # Local OpenAI-compatible servers ignore the key but the client requires
        # a non-empty value; never read a real OPENAI_API_KEY from the env.
        api_key=Secret.from_token("not-needed-for-local-server"),
        model=settings.llm_model,
        api_base_url=settings.llm_base_url,
        generation_kwargs={"temperature": 0},  # §7.7 reproducible answers
        timeout=settings.llm_timeout,
    )

    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("generator", generator)

    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "generator.prompt")
    return pipeline


@lru_cache(maxsize=1)
def _default_pipeline() -> Pipeline:
    """Process-wide query pipeline (built once; loads the embedder + store)."""
    from corpus_rag.document_store import build_document_store

    settings = get_settings()
    return build_query_pipeline(build_document_store(settings), settings)


def run_query(
    query: str,
    *,
    pipeline: Pipeline | None = None,
    settings: Settings | None = None,
) -> tuple[str, list[Document]]:
    """Run a grounded query (root §4.3, §2A).

    :param query: The user's question.
    :param pipeline: Override pipeline (defaults to the cached process pipeline).
    :param settings: Override settings (defaults to cached process settings).
    :returns: ``(answer, documents)`` — documents in cosine-ranked order, always
        returned even on abstention.
    """
    settings = settings or get_settings()
    pipeline = pipeline or _default_pipeline()

    # Empty/whitespace query → undefined embedding behavior; abstain immediately.
    if not query.strip():
        return ABSTENTION_ANSWER, []

    # NOTE: the generator runs unconditionally as part of pipeline.run — even when
    # the MIN_SCORE gate below will abstain and discard its reply. Acceptable for
    # this MVP slice; a retriever-only pre-pass would avoid the wasted LLM call.
    result = pipeline.run(
        {"text_embedder": {"text": query}, "prompt_builder": {"query": query}},
        include_outputs_from={"retriever"},
    )

    documents: list[Document] = result["retriever"]["documents"]

    # Grounding gate (§2A.3): drop chunks below the MIN_SCORE floor; abstain when
    # nothing remains, discarding any generated prose to avoid contamination.
    if settings.min_score > 0.0:
        documents = [d for d in documents if (d.score or 0.0) >= settings.min_score]

    if not documents:
        return ABSTENTION_ANSWER, documents

    replies = result.get("generator", {}).get("replies") or []
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
    """One retrieved chunk with both its cosine and rerank rank/score."""

    document: Document
    cosine_rank: int
    cosine_score: float | None
    rerank_rank: int
    rerank_score: float | None


@dataclass
class RerankEngine:
    """Warmed components for a rerank query (built once; heavy models)."""

    text_embedder: SentenceTransformersTextEmbedder
    retriever: PgvectorEmbeddingRetriever
    ranker: TransformersSimilarityRanker
    prompt_builder: PromptBuilder
    generator: OpenAIGenerator


def build_rerank_engine(
    document_store: PgvectorDocumentStore,
    settings: Settings | None = None,
) -> RerankEngine:
    """Build + warm up the components for cosine-retrieve → cross-encoder rerank."""
    settings = settings or get_settings()

    text_embedder = SentenceTransformersTextEmbedder(model=settings.embed_model_id)
    text_embedder.warm_up()
    retriever = PgvectorEmbeddingRetriever(
        document_store=document_store,
        top_k=settings.rerank_candidates,
    )
    ranker = TransformersSimilarityRanker(
        model=settings.rerank_model_id,
        top_k=settings.rerank_candidates,
    )
    ranker.warm_up()
    prompt_builder = PromptBuilder(
        template=RAG_PROMPT_TEMPLATE,
        required_variables=["query", "documents"],
    )
    generator = OpenAIGenerator(
        api_key=Secret.from_token("not-needed-for-local-server"),
        model=settings.llm_model,
        api_base_url=settings.llm_base_url,
        generation_kwargs={"temperature": 0},
        timeout=settings.llm_timeout,
    )
    return RerankEngine(text_embedder, retriever, ranker, prompt_builder, generator)


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

    # Snapshot cosine rank+score NOW; the ranker overwrites Document.score.
    cosine_by_id = {
        doc.id: (rank, doc.score) for rank, doc in enumerate(cosine_docs, start=1)
    }

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

    prompt = engine.prompt_builder.run(
        query=query, documents=[rs.document for rs in grounded]
    )["prompt"]
    replies = engine.generator.run(prompt=prompt).get("replies") or []
    if not replies:
        logger.warning("Generator returned no replies for query %r", query)
        return ABSTENTION_ANSWER, ranked_sources
    return replies[0], ranked_sources
