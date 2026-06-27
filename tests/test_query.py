"""Tests for the query pipeline + §2A grounding contract.

Offline tests cover the prompt template and the ``run_query`` grounding gate
with a mock pipeline (no models, DB, or LLM). Live tests (``@pytest.mark.live``)
run the real pipeline against the ingested corpus + local LLM and cover §7.5,
§7.7, and the §2A A1/A2 checks; they self-skip when the corpus is empty.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import pytest
from haystack import Document

from corpus_rag.pipelines.query import RankedSource, run_query, run_query_reranked
from corpus_rag.prompts import ABSTENTION_ANSWER, RAG_PROMPT_TEMPLATE
from corpus_rag.settings import Settings


def _settings(**overrides) -> Settings:
    base = dict(min_score=0.0, top_k=10)
    base.update(overrides)
    return Settings(_env_file=None, **base)


def _mock_pipeline(documents: list[Document], reply: str = "grounded answer"):
    pipeline = MagicMock()
    pipeline.run.return_value = {
        "retriever": {"documents": documents},
        "generator": {"replies": [reply]},
    }
    return pipeline


# --- prompt template -----------------------------------------------------


def test_prompt_template_enforces_grounding() -> None:
    t = RAG_PROMPT_TEMPLATE
    assert ABSTENTION_ANSWER in t
    assert "ONLY the RETRIEVED" in t
    assert "NEVER use specific clinical knowledge" in t
    # Jinja hooks the builder fills in.
    assert "{{ query }}" in t
    assert "doc.content" in t


# --- run_query grounding gate -------------------------------------------


def test_run_query_returns_answer_and_ranked_docs() -> None:
    docs = [Document(content="a", score=0.9), Document(content="b", score=0.5)]
    pipeline = _mock_pipeline(docs, reply="from sources")

    answer, returned = run_query("q", pipeline=pipeline, settings=_settings())

    assert answer == "from sources"
    assert returned == docs  # order preserved, no re-sort


@pytest.mark.parametrize("q", ["", "   ", "\n\t"])
def test_run_query_empty_query_abstains_without_running_pipeline(q: str) -> None:
    pipeline = _mock_pipeline([Document(content="a", score=0.9)])

    answer, returned = run_query(q, pipeline=pipeline, settings=_settings())

    assert answer == ABSTENTION_ANSWER
    assert returned == []
    pipeline.run.assert_not_called()


def test_run_query_abstains_on_empty_retrieval() -> None:
    pipeline = _mock_pipeline([], reply="should be discarded")

    answer, returned = run_query("q", pipeline=pipeline, settings=_settings())

    assert answer == ABSTENTION_ANSWER
    assert returned == []


def test_run_query_min_score_filters_then_abstains() -> None:
    docs = [Document(content="a", score=0.2), Document(content="b", score=0.1)]
    pipeline = _mock_pipeline(docs, reply="should be discarded")

    answer, returned = run_query("q", pipeline=pipeline, settings=_settings(min_score=0.5))

    assert answer == ABSTENTION_ANSWER
    assert returned == []


def test_run_query_min_score_keeps_grounded_docs() -> None:
    docs = [Document(content="a", score=0.9), Document(content="b", score=0.1)]
    pipeline = _mock_pipeline(docs, reply="grounded")

    answer, returned = run_query("q", pipeline=pipeline, settings=_settings(min_score=0.5))

    assert answer == "grounded"
    assert [d.content for d in returned] == ["a"]


def test_run_query_abstains_when_generator_empty() -> None:
    docs = [Document(content="a", score=0.9)]
    pipeline = MagicMock()
    pipeline.run.return_value = {"retriever": {"documents": docs}, "generator": {"replies": []}}

    answer, returned = run_query("q", pipeline=pipeline, settings=_settings())

    assert answer == ABSTENTION_ANSWER
    assert returned == docs


# --- reranking -----------------------------------------------------------


def _rerank_engine(cosine_docs, reranked_with_scores, reply="grounded answer"):
    """Fake RerankEngine: retriever yields cosine_docs; ranker reorders + rescores.

    ``reranked_with_scores`` is a list of (document, rerank_score) in rerank order.
    The ranker mutates ``document.score`` in place (as the real cross-encoder
    does), so run_query_reranked must have snapshotted the cosine scores first.
    """
    engine = MagicMock()
    engine.text_embedder.run.return_value = {"embedding": [0.0, 0.1, 0.2]}
    engine.retriever.run.return_value = {"documents": cosine_docs}

    def _rerank(query, documents):
        ordered = []
        for doc, score in reranked_with_scores:
            # The real TransformersSimilarityRanker mutates Document.score in
            # place; the fake mirrors that (so the snapshot-before-rerank logic is
            # actually exercised). Haystack warns on the mutation — silence the
            # cosmetic warning here rather than weaken the fake.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mutating attribute")
                doc.score = score
            ordered.append(doc)
        return {"documents": ordered}

    engine.ranker.run.side_effect = _rerank
    engine.prompt_builder.run.return_value = {"prompt": "PROMPT"}
    engine.generator.run.return_value = {"replies": [reply]}
    return engine


def test_rerank_overrides_cosine_order_and_keeps_both_scores() -> None:
    a = Document(content="aaa", score=0.9)
    b = Document(content="bbb", score=0.8)
    c = Document(content="ccc", score=0.7)
    # Cross-encoder promotes c (cosine #3) to rerank #1, demoting a and b.
    engine = _rerank_engine([a, b, c], [(c, 0.99), (a, 0.50), (b, 0.10)], reply="from sources")

    answer, sources = run_query_reranked("q", engine=engine, settings=_settings())

    assert answer == "from sources"
    assert all(isinstance(s, RankedSource) for s in sources)
    # Rerank order.
    assert [s.document.id for s in sources] == [c.id, a.id, b.id]
    assert [s.rerank_rank for s in sources] == [1, 2, 3]
    assert [s.rerank_score for s in sources] == [0.99, 0.50, 0.10]
    # Original cosine rank/score preserved per chunk (the override is visible).
    assert [s.cosine_rank for s in sources] == [3, 1, 2]
    assert [s.cosine_score for s in sources] == [0.7, 0.9, 0.8]


def test_rerank_grounds_llm_on_top_k_only_but_returns_all_candidates() -> None:
    a = Document(content="aaa", score=0.9)
    b = Document(content="bbb", score=0.8)
    c = Document(content="ccc", score=0.7)
    engine = _rerank_engine([a, b, c], [(c, 0.99), (a, 0.50), (b, 0.10)], reply="from sources")

    # top_k=2: LLM sees only the top 2 reranked (c, a); UI still gets all 3.
    _, sources = run_query_reranked("q", engine=engine, settings=_settings(top_k=2))

    assert len(sources) == 3  # all candidates surfaced for display
    sent = engine.prompt_builder.run.call_args.kwargs["documents"]
    assert [d.id for d in sent] == [c.id, a.id]  # only top_k, in rerank order


def test_rerank_empty_query_abstains_without_running_engine() -> None:
    engine = _rerank_engine([Document(content="a", score=0.9)], [])
    answer, sources = run_query_reranked("  ", engine=engine, settings=_settings())

    assert answer == ABSTENTION_ANSWER
    assert sources == []
    engine.text_embedder.run.assert_not_called()


def test_rerank_min_score_gate_abstains_but_still_returns_sources() -> None:
    a = Document(content="aaa", score=0.2)
    b = Document(content="bbb", score=0.1)
    engine = _rerank_engine([a, b], [(b, 0.95), (a, 0.30)], reply="discarded")

    answer, sources = run_query_reranked("q", engine=engine, settings=_settings(min_score=0.5))

    # Cosine scores below the floor -> abstain, but the 2 sources still surface.
    assert answer == ABSTENTION_ANSWER
    assert [s.document.id for s in sources] == [b.id, a.id]
    engine.generator.run.assert_not_called()


# --- live ----------------------------------------------------------------


def _live_pipeline_and_store():
    from corpus_rag.document_store import build_document_store
    from corpus_rag.pipelines.query import build_query_pipeline
    from corpus_rag.settings import get_settings

    settings = get_settings()
    store = build_document_store(settings)
    if store.count_documents() == 0:
        pytest.skip("Empty corpus; ingest a sample corpus first (live).")
    return build_query_pipeline(store, settings), store, settings


@pytest.mark.live
def test_live_retrieval_count_and_ordering() -> None:
    """§7.5: min(TOP_K, N) docs returned with non-increasing scores."""
    pipeline, store, settings = _live_pipeline_and_store()
    if settings.min_score > 0.0:
        pytest.skip("MIN_SCORE>0 post-filters docs; count assertion not applicable.")
    _, docs = run_query("What does the guideline recommend?", pipeline=pipeline, settings=settings)

    expected = min(settings.top_k, store.count_documents())
    assert len(docs) == expected
    scores = [d.score for d in docs]
    assert all(a >= b for a, b in zip(scores, scores[1:], strict=False))


@pytest.mark.live
def test_live_retrieval_order_is_deterministic() -> None:
    """§7.7: same query + corpus yields a stable retrieval order."""
    pipeline, _, settings = _live_pipeline_and_store()
    q = "What does the guideline recommend?"
    _, first = run_query(q, pipeline=pipeline, settings=settings)
    _, second = run_query(q, pipeline=pipeline, settings=settings)

    assert [d.id for d in first] == [d.id for d in second]


@pytest.mark.live
def test_live_a2_non_abstain_answer_has_verbatim_source() -> None:
    """§2A A2: a grounded answer is accompanied by >=1 verbatim source chunk."""
    pipeline, store, settings = _live_pipeline_and_store()
    answer, docs = run_query(
        "Which oral antibiotic is recommended as the first-line treatment for pneumonia in adults?",
        pipeline=pipeline,
        settings=settings,
    )

    assert answer != ABSTENTION_ANSWER
    assert len(docs) >= 1
    # Retrieved content is the stored verbatim chunk (same field), byte-equal.
    stored = {d.id: d.content for d in store.filter_documents()}
    assert docs[0].content == stored[docs[0].id]
    assert docs[0].content.strip()


@pytest.mark.live
def test_live_a1_no_match_abstains() -> None:
    """§2A A1: with no grounding above the floor, abstain (no parametric claim)."""
    pipeline, _, settings = _live_pipeline_and_store()
    # Force the no-grounding path via a high MIN_SCORE floor.
    strict = settings.model_copy(update={"min_score": 0.999})
    answer, _ = run_query("zzzz unrelated nonsense qqqq", pipeline=pipeline, settings=strict)

    assert answer == ABSTENTION_ANSWER


# --- live reranking ------------------------------------------------------


def _live_rerank_engine_and_store():
    from corpus_rag.document_store import build_document_store
    from corpus_rag.pipelines.query import build_rerank_engine
    from corpus_rag.settings import get_settings

    settings = get_settings()
    store = build_document_store(settings)
    if store.count_documents() == 0:
        pytest.skip("Empty corpus; ingest a sample corpus first (live).")
    return build_rerank_engine(store, settings), store, settings


@pytest.mark.live
def test_live_rerank_returns_candidates_with_both_rankings() -> None:
    """Top RERANK_CANDIDATES surfaced, each with cosine + rerank rank/score."""
    engine, store, settings = _live_rerank_engine_and_store()
    _, sources = run_query_reranked(
        "What does the guideline recommend?", engine=engine, settings=settings
    )

    expected = min(settings.rerank_candidates, store.count_documents())
    assert len(sources) == expected
    # Rerank ranks are 1..N contiguous; both score channels populated.
    assert [s.rerank_rank for s in sources] == list(range(1, len(sources) + 1))
    assert {s.cosine_rank for s in sources} == set(range(1, len(sources) + 1))
    assert all(s.cosine_score is not None and s.rerank_score is not None for s in sources)
    # Rerank scores are non-increasing (ranker sorts by relevance).
    rs = [s.rerank_score for s in sources]
    assert all(a >= b for a, b in zip(rs, rs[1:], strict=False))


@pytest.mark.live
def test_live_rerank_overrides_cosine_order() -> None:
    """Empirical demo (NOT a hard contract): the rerank reorders the cosine list.

    This asserts the cross-encoder disagrees with cosine for this query/corpus.
    It is a demonstration, not a property of the implementation — on a tiny or
    trivially-separable corpus the two orders can legitimately agree, so we skip
    rather than fail when there are too few sources to expect disagreement.
    """
    engine, _, settings = _live_rerank_engine_and_store()
    _, sources = run_query_reranked(
        "Which oral antibiotic is recommended as the first-line treatment for pneumonia in adults?",
        engine=engine,
        settings=settings,
    )

    if len(sources) < 5:
        pytest.skip("Too few sources to expect a rerank/cosine disagreement.")
    # If rerank changed nothing, cosine_rank would equal rerank_rank for all.
    cosine_order = [s.cosine_rank for s in sources]
    assert cosine_order != sorted(cosine_order), "rerank did not reorder cosine list"
