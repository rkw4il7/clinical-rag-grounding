"""Tests for the query pipeline + §2A grounding contract.

Offline tests cover the prompt template and the ``run_query`` grounding gate
with a mock pipeline (no models, DB, or LLM). Live tests (``@pytest.mark.live``)
run the real pipeline against the ingested corpus + local LLM and cover §7.5,
§7.7, and the §2A A1/A2 checks; they self-skip when the corpus is empty.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from haystack import Document

from corpus_rag.pipelines.query import run_query
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
    _, docs = run_query(
        "What does the guideline recommend?", pipeline=pipeline, settings=settings
    )

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
        "Which oral antibiotic is recommended as the first-line treatment "
        "for pneumonia in adults?",
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
    answer, _ = run_query(
        "zzzz unrelated nonsense qqqq", pipeline=pipeline, settings=strict
    )

    assert answer == ABSTENTION_ANSWER
