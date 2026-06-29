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
from collections.abc import Callable
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

from corpus_rag.prompts import ABSTENTION_ANSWER, CONTINUE_RAG_PROMPT_TEMPLATE, RAG_PROMPT_TEMPLATE
from corpus_rag.settings import Settings, get_settings

if TYPE_CHECKING:
    from haystack import Document
    from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

logger = logging.getLogger(__name__)

QueryProgress = Callable[[str], None]
GenerationProgress = Callable[[str], None]
FinishReasonCallback = Callable[[str | None], None]

# Continuation tuning. The grounded source chunks (unchanged across rounds) carry
# the answer's content; only a short TAIL of the answer-so-far is needed as the
# seam to continue from. Resending the WHOLE growing answer each round made prompt
# prefill O(answer^2) and was a primary cause of slow multi-round responses.
_CONTINUATION_TAIL_CHARS = 600
# If a length-truncated round emits less than this, the context is effectively
# saturated — further rounds resend an even bigger prompt for ever-smaller gains,
# so stop instead of grinding to the round cap.
_MIN_CONTINUATION_PROGRESS_CHARS = 32


def _generation_callbacks(
    generation_progress: GenerationProgress | None,
) -> tuple[Callable | None, list[str]]:
    streamed_chunks: list[str] = []
    if generation_progress is None:
        return None, streamed_chunks

    def streaming_callback(chunk) -> None:
        content = chunk.content or ""
        if content:
            streamed_chunks.append(content)
            generation_progress(content)

    return streaming_callback, streamed_chunks


def _first_finish_reason(result: dict) -> str | None:
    return (result.get("meta") or [{}])[0].get("finish_reason")


def _generated_reply(result: dict, streamed_chunks: list[str]) -> str:
    """Choose the most complete nonblank generated text available."""
    candidates: list[str] = []
    streamed = "".join(streamed_chunks)
    if streamed.strip():
        candidates.append(streamed)
    for reply in result.get("replies") or []:
        if isinstance(reply, str) and reply.strip():
            candidates.append(reply)
    if not candidates:
        return ""
    return max(candidates, key=len)


@lru_cache(maxsize=1)
def _continuation_prompt_builder() -> PromptBuilder:
    """Cached builder for the continuation template (Jinja, not str.replace).

    Rendering through Jinja inserts source/query/partial-answer text as literal
    variable VALUES, so corpus or query text containing a placeholder token can
    never corrupt the prompt — closes the str.replace injection vector.
    """
    return PromptBuilder(
        template=CONTINUE_RAG_PROMPT_TEMPLATE,
        required_variables=["documents", "query", "partial_answer"],
    )


def _continue_from_docs(
    engine,
    query: str,
    partial_answer: str,
    grounded_docs: list[Document],
    *,
    generation_progress: GenerationProgress | None = None,
) -> tuple[str, str | None]:
    """Generate one continuation segment grounded in the SAME source chunks.

    Returns ``(continuation_text, finish_reason)``. Deliberately reuses the
    already-grounded chunks (no fresh retrieval) so the continuation's safety
    boundary is identical to the original answer's.
    """
    # Only the tail of the answer-so-far is needed as the continuation seam; the
    # sources (unchanged) carry the grounding. Keeps the prompt bounded per round.
    seam = partial_answer[-_CONTINUATION_TAIL_CHARS:]
    prompt = _continuation_prompt_builder().run(
        documents=grounded_docs, query=query, partial_answer=seam
    )["prompt"]
    streaming_callback, streamed_chunks = _generation_callbacks(generation_progress)
    result = engine.generator.run(prompt=prompt, streaming_callback=streaming_callback)
    return _generated_reply(result, streamed_chunks), _first_finish_reason(result)


def _complete_truncated_answer(
    engine,
    query: str,
    answer: str,
    finish_reason: str | None,
    grounded_docs: list[Document],
    *,
    max_rounds: int,
    progress: QueryProgress | None = None,
    generation_progress: GenerationProgress | None = None,
) -> tuple[str, str | None]:
    """Loop continuation turns until the model stops naturally or the cap is hit.

    A length-truncated answer (``finish_reason == "length"``) is continued from
    the same grounded chunks, appending each segment, until the generator reports
    a non-length stop or ``max_rounds`` is reached. Transparent to the caller:
    only the final, stitched answer and its final finish reason are returned.
    """
    rounds = 0
    while finish_reason == "length" and rounds < max_rounds:
        rounds += 1
        if progress:
            progress(f"Extending response (continuation {rounds} of up to {max_rounds})")
        continuation, finish_reason = _continue_from_docs(
            engine, query, answer, grounded_docs, generation_progress=generation_progress
        )
        if not continuation:
            # Nothing more produced; stop rather than spin to the cap.
            break
        answer += continuation
        # Non-progress guard: still length-truncated but barely any new text means
        # the context is saturated — continuing only gets slower. Stop early.
        progressed = len(continuation.strip())
        if finish_reason == "length" and progressed < _MIN_CONTINUATION_PROGRESS_CHARS:
            logger.warning(
                "Continuation produced only %d chars while still length-truncated "
                "for query %r; stopping (context likely saturated).",
                progressed,
                query,
            )
            break
    if finish_reason == "length" and rounds >= max_rounds:
        logger.warning(
            "Answer still truncated after %d continuation round(s) for query %r; "
            "raise LLM_MAX_TOKENS or MAX_CONTINUATION_ROUNDS.",
            max_rounds,
            query,
        )
    return answer, finish_reason


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
        generation_kwargs={
            "temperature": 0,  # §7.7 reproducible answers
            "max_tokens": settings.llm_max_tokens,
        },
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
    progress: QueryProgress | None = None,
    generation_progress: GenerationProgress | None = None,
    finish_reason_callback: FinishReasonCallback | None = None,
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

    if progress:
        progress("Embedding query")
    embedding = engine.text_embedder.run(text=query)["embedding"]
    if progress:
        progress("Retrieving source chunks")
    documents: list[Document] = engine.retriever.run(query_embedding=embedding)["documents"]

    if progress:
        progress("Applying grounding gate")
    # Grounding gate (§2A.3): drop chunks below the MIN_SCORE floor; abstain when
    # nothing remains — BEFORE the generator runs, so no ungrounded prose is ever
    # produced (not merely discarded).
    if settings.min_score > 0.0:
        documents = [d for d in documents if (d.score or 0.0) >= settings.min_score]

    if not documents:
        return ABSTENTION_ANSWER, documents

    if progress:
        progress("Building grounded prompt")
    prompt = engine.prompt_builder.run(query=query, documents=documents)["prompt"]
    if progress:
        progress("Generating response")
    streaming_callback, streamed_chunks = _generation_callbacks(generation_progress)
    result = engine.generator.run(
        prompt=prompt,
        streaming_callback=streaming_callback,
    )
    reply = _generated_reply(result, streamed_chunks)
    if not reply:
        # Distinct from a grounding abstention: the LLM produced nothing (network
        # error, context overflow, refusal). Same return, but make it detectable.
        logger.warning("Generator returned no replies for query %r", query)
        if finish_reason_callback:
            finish_reason_callback(_first_finish_reason(result))
        return ABSTENTION_ANSWER, documents

    # Transparently finish a length-truncated answer from the same grounded
    # chunks (parity with run_query_reranked), bounded by the round cap.
    answer, finish_reason = _complete_truncated_answer(
        engine,
        query,
        reply,
        _first_finish_reason(result),
        documents,
        max_rounds=settings.max_continuation_rounds,
        progress=progress,
        generation_progress=generation_progress,
    )
    if finish_reason_callback:
        finish_reason_callback(finish_reason)
    return answer, documents


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
    progress: QueryProgress | None = None,
    generation_progress: GenerationProgress | None = None,
    finish_reason_callback: FinishReasonCallback | None = None,
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

    if progress:
        progress("Embedding query")
    embedding = engine.text_embedder.run(text=query)["embedding"]
    if progress:
        progress("Retrieving candidate chunks")
    cosine_docs = engine.retriever.run(query_embedding=embedding)["documents"]

    # Snapshot cosine rank+score NOW (the ranker overwrites Document.score). Key
    # by Document.id, NOT Python object identity: the ranker returns NEW Document
    # objects (copies), so id() would not match the cosine snapshot. Document.id
    # is a content+meta hash; the pgvector store keys rows by id (OVERWRITE), so
    # the retrieved set has unique ids — no collision to worry about here.
    cosine_by_id = {doc.id: (rank, doc.score) for rank, doc in enumerate(cosine_docs, start=1)}

    if progress:
        progress(f"Reranking {len(cosine_docs)} candidate chunk(s)")
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

    if progress:
        progress("Applying grounding gate")
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

    if progress:
        progress("Building grounded prompt")
    prompt = engine.prompt_builder.run(query=query, documents=[rs.document for rs in llm_sources])[
        "prompt"
    ]
    if progress:
        progress("Generating response")
    streaming_callback, streamed_chunks = _generation_callbacks(generation_progress)
    result = engine.generator.run(
        prompt=prompt,
        streaming_callback=streaming_callback,
    )
    reply = _generated_reply(result, streamed_chunks)
    if not reply:
        logger.warning("Generator returned no replies for query %r", query)
        if finish_reason_callback:
            finish_reason_callback(_first_finish_reason(result))
        return ABSTENTION_ANSWER, ranked_sources

    # The summary of a finite slug of sources is itself finite: if the backend
    # truncated on length, transparently continue from the SAME grounded chunks
    # (no fresh retrieval) until it stops naturally or the round cap is hit.
    answer, finish_reason = _complete_truncated_answer(
        engine,
        query,
        reply,
        _first_finish_reason(result),
        [rs.document for rs in llm_sources],
        max_rounds=settings.max_continuation_rounds,
        progress=progress,
        generation_progress=generation_progress,
    )
    if finish_reason_callback:
        finish_reason_callback(finish_reason)
    return answer, ranked_sources


def continue_reranked_answer(
    query: str,
    partial_answer: str,
    ranked_sources: list[RankedSource],
    *,
    engine: RerankEngine | None = None,
    generation_progress: GenerationProgress | None = None,
    finish_reason_callback: FinishReasonCallback | None = None,
) -> str:
    """Continue a length-truncated answer using the same grounded source chunks.

    This is deliberately not a fresh retrieval. The continuation prompt receives
    only the chunks already marked ``used_for_grounding`` so the safety boundary
    remains the same as the original answer.
    """
    engine = engine or _default_rerank_engine()
    grounded = [rs.document for rs in ranked_sources if rs.used_for_grounding]
    if not grounded:
        return ""

    reply, finish_reason = _continue_from_docs(
        engine, query, partial_answer, grounded, generation_progress=generation_progress
    )
    if finish_reason_callback:
        finish_reason_callback(finish_reason)
    if not reply:
        logger.warning("Generator returned no continuation for query %r", query)
        return ""
    return reply
