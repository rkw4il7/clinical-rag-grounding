"""Tests for the query pipeline + §2A grounding contract.

Offline tests cover the prompt template and the ``run_query`` grounding gate
with a mock pipeline (no models, DB, or LLM). Live tests (``@pytest.mark.live``)
run the real pipeline against the ingested corpus + local LLM and cover §7.5,
§7.7, and the §2A A1/A2 checks; they self-skip when the corpus is empty.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from haystack import Document

from corpus_rag.pipelines.query import (
    RankedSource,
    continue_reranked_answer,
    run_query,
    run_query_reranked,
)
from corpus_rag.prompts import ABSTENTION_ANSWER, RAG_PROMPT_TEMPLATE
from corpus_rag.settings import Settings


def _settings(**overrides) -> Settings:
    base = dict(min_score=0.0, top_k=10)
    base.update(overrides)
    return Settings(_env_file=None, **base)


def _mock_engine(documents: list[Document], reply: str = "grounded answer"):
    """Fake QueryEngine driving embed -> retrieve -> gate -> generate stepwise."""
    engine = MagicMock()
    engine.text_embedder.run.return_value = {"embedding": [0.0, 0.1, 0.2]}
    engine.retriever.run.return_value = {"documents": documents}
    engine.prompt_builder.run.return_value = {"prompt": "PROMPT"}
    engine.generator.run.return_value = {"replies": [reply]}
    return engine


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
    engine = _mock_engine(docs, reply="from sources")

    answer, returned = run_query("q", engine=engine, settings=_settings())

    assert answer == "from sources"
    assert returned == docs  # order preserved, no re-sort


@pytest.mark.parametrize("q", ["", "   ", "\n\t"])
def test_run_query_empty_query_abstains_without_running_engine(q: str) -> None:
    engine = _mock_engine([Document(content="a", score=0.9)])

    answer, returned = run_query(q, engine=engine, settings=_settings())

    assert answer == ABSTENTION_ANSWER
    assert returned == []
    engine.text_embedder.run.assert_not_called()


def test_run_query_abstains_on_empty_retrieval_without_generating() -> None:
    engine = _mock_engine([], reply="should never be produced")

    answer, returned = run_query("q", engine=engine, settings=_settings())

    assert answer == ABSTENTION_ANSWER
    assert returned == []
    # Gate-before-generate: the LLM is never called without grounding.
    engine.generator.run.assert_not_called()


def test_run_query_min_score_filters_then_abstains_without_generating() -> None:
    docs = [Document(content="a", score=0.2), Document(content="b", score=0.1)]
    engine = _mock_engine(docs, reply="should never be produced")

    answer, returned = run_query("q", engine=engine, settings=_settings(min_score=0.5))

    assert answer == ABSTENTION_ANSWER
    assert returned == []
    # Sub-floor grounding → abstain BEFORE generation, not discard after.
    engine.generator.run.assert_not_called()


def test_run_query_min_score_keeps_grounded_docs() -> None:
    docs = [Document(content="a", score=0.9), Document(content="b", score=0.1)]
    engine = _mock_engine(docs, reply="grounded")

    answer, returned = run_query("q", engine=engine, settings=_settings(min_score=0.5))

    assert answer == "grounded"
    assert [d.content for d in returned] == ["a"]
    engine.generator.run.assert_called_once()


def test_run_query_abstains_when_generator_empty() -> None:
    docs = [Document(content="a", score=0.9)]
    engine = _mock_engine(docs)
    engine.generator.run.return_value = {"replies": []}

    answer, returned = run_query("q", engine=engine, settings=_settings())

    assert answer == ABSTENTION_ANSWER
    assert returned == docs


def test_run_query_auto_continues_truncated_answer() -> None:
    """run_query has parity with run_query_reranked: length truncation continues."""
    docs = [Document(content="a", score=0.9)]
    engine = _mock_engine(docs, reply="unused")
    engine.generator.run.side_effect = [
        {"replies": ["part one "], "meta": [{"finish_reason": "length"}]},
        {"replies": ["part two."], "meta": [{"finish_reason": "stop"}]},
    ]
    finish_reasons = []

    answer, _ = run_query(
        "q", engine=engine, settings=_settings(), finish_reason_callback=finish_reasons.append
    )

    assert answer == "part one part two."
    assert finish_reasons == ["stop"]
    assert engine.generator.run.call_count == 2


# --- reranking -----------------------------------------------------------


def _rerank_engine(cosine_docs, reranked_with_scores, reply="grounded answer"):
    """Fake RerankEngine: retriever yields cosine_docs; ranker reorders + rescores.

    ``reranked_with_scores`` is a list of (document, rerank_score) in rerank order.
    The real TransformersSimilarityRanker returns NEW Document objects (copies
    with the rerank score), preserving each ``Document.id``. The fake mirrors that
    with ``dataclasses.replace`` — NOT in-place mutation — so run_query_reranked is
    forced to map the cosine snapshot back by ``Document.id`` (object identity
    would not survive the copy). Guards the id-vs-identity keying regression.
    """
    engine = MagicMock()
    engine.text_embedder.run.return_value = {"embedding": [0.0, 0.1, 0.2]}
    engine.retriever.run.return_value = {"documents": cosine_docs}

    def _rerank(query, documents):
        return {"documents": [replace(doc, score=score) for doc, score in reranked_with_scores]}

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
    # used_for_grounding marks EXACTLY the chunks fed to the LLM (rerank order c,a,b).
    assert [s.used_for_grounding for s in sources] == [True, True, False]


def test_rerank_reports_progress_steps() -> None:
    a = Document(content="aaa", score=0.9)
    b = Document(content="bbb", score=0.8)
    engine = _rerank_engine([a, b], [(b, 0.95), (a, 0.30)], reply="from sources")

    steps = []
    run_query_reranked("q", engine=engine, settings=_settings(), progress=steps.append)

    assert steps == [
        "Embedding query",
        "Retrieving candidate chunks",
        "Reranking 2 candidate chunk(s)",
        "Applying grounding gate",
        "Building grounded prompt",
        "Generating response",
    ]


def test_rerank_forwards_generation_stream_chunks() -> None:
    a = Document(content="aaa", score=0.9)
    engine = _rerank_engine([a], [(a, 0.95)], reply="from sources")

    def _generate(prompt, streaming_callback=None):
        assert prompt == "PROMPT"
        streaming_callback(type("Chunk", (), {"content": "from "})())
        streaming_callback(type("Chunk", (), {"content": "sources"})())
        return {"replies": ["from sources"], "meta": [{"finish_reason": "stop"}]}

    engine.generator.run.side_effect = _generate
    chunks = []

    answer, _ = run_query_reranked(
        "q",
        engine=engine,
        settings=_settings(),
        generation_progress=chunks.append,
    )

    assert answer == "from sources"
    assert chunks == ["from ", "sources"]


def test_rerank_uses_streamed_text_when_final_reply_is_empty() -> None:
    a = Document(content="aaa", score=0.9)
    engine = _rerank_engine([a], [(a, 0.95)], reply="unused")

    def _generate(prompt, streaming_callback=None):
        assert prompt == "PROMPT"
        streaming_callback(type("Chunk", (), {"content": "streamed "})())
        streaming_callback(type("Chunk", (), {"content": "answer"})())
        return {"replies": [""], "meta": [{"finish_reason": "stop"}]}

    engine.generator.run.side_effect = _generate

    answer, sources = run_query_reranked(
        "q",
        engine=engine,
        settings=_settings(),
        generation_progress=lambda _text: None,
    )

    assert answer == "streamed answer"
    assert sources[0].used_for_grounding is True


def test_rerank_empty_final_reply_without_stream_abstains() -> None:
    a = Document(content="aaa", score=0.9)
    engine = _rerank_engine([a], [(a, 0.95)], reply="unused")
    engine.generator.run.return_value = {"replies": [""], "meta": [{"finish_reason": "stop"}]}

    answer, sources = run_query_reranked("q", engine=engine, settings=_settings())

    assert answer == ABSTENTION_ANSWER
    assert sources[0].used_for_grounding is True


def test_rerank_prefers_streamed_text_over_shorter_final_reply() -> None:
    a = Document(content="aaa", score=0.9)
    engine = _rerank_engine([a], [(a, 0.95)], reply="unused")

    def _generate(prompt, streaming_callback=None):
        assert prompt == "PROMPT"
        streaming_callback(type("Chunk", (), {"content": "complete streamed answer."})())
        return {"replies": ["complete streamed"], "meta": [{"finish_reason": "stop"}]}

    engine.generator.run.side_effect = _generate

    answer, _ = run_query_reranked(
        "q",
        engine=engine,
        settings=_settings(),
        generation_progress=lambda _text: None,
    )

    assert answer == "complete streamed answer."


def test_rerank_reports_finish_reason() -> None:
    a = Document(content="aaa", score=0.9)
    engine = _rerank_engine([a], [(a, 0.95)], reply="from sources")
    engine.generator.run.return_value = {
        "replies": ["from sources"],
        "meta": [{"finish_reason": "stop"}],
    }
    finish_reasons = []

    run_query_reranked(
        "q",
        engine=engine,
        settings=_settings(),
        finish_reason_callback=finish_reasons.append,
    )

    assert finish_reasons == ["stop"]


def test_rerank_auto_continues_until_model_stops() -> None:
    """A length-truncated answer is transparently continued, then stitched."""
    a = Document(content="aaa", score=0.9)
    engine = _rerank_engine([a], [(a, 0.95)], reply="unused")
    engine.generator.run.side_effect = [
        {"replies": ["part one "], "meta": [{"finish_reason": "length"}]},
        {"replies": ["part two."], "meta": [{"finish_reason": "stop"}]},
    ]
    finish_reasons = []

    answer, _ = run_query_reranked(
        "q",
        engine=engine,
        settings=_settings(),
        finish_reason_callback=finish_reasons.append,
    )

    assert answer == "part one part two."  # stitched across turns
    assert finish_reasons == ["stop"]  # only the FINAL reason is reported
    assert engine.generator.run.call_count == 2
    # Continuation grounds on the same chunk via the continuation template.
    cont_prompt = engine.generator.run.call_args.kwargs["prompt"]
    assert "part one" in cont_prompt  # prior answer fed back as PARTIAL ANSWER


def test_rerank_auto_continue_respects_round_cap() -> None:
    """Persistent length truncation stops at MAX_CONTINUATION_ROUNDS, not forever."""
    a = Document(content="aaa", score=0.9)
    engine = _rerank_engine([a], [(a, 0.95)], reply="unused")
    # Each round emits a healthy chunk (above the non-progress threshold) so the
    # cap — not the saturation guard — is what stops the loop.
    seg = "a meaningful continuation segment of grounded prose. "
    engine.generator.run.return_value = {
        "replies": [seg],
        "meta": [{"finish_reason": "length"}],
    }
    finish_reasons = []

    answer, _ = run_query_reranked(
        "q",
        engine=engine,
        settings=_settings(max_continuation_rounds=2),
        finish_reason_callback=finish_reasons.append,
    )

    # Initial generation + exactly 2 continuation rounds (the cap).
    assert engine.generator.run.call_count == 3
    assert answer == seg * 3
    assert finish_reasons == ["length"]  # still truncated; surfaced to the UI


def test_rerank_auto_continue_stops_on_non_progress() -> None:
    """A length-truncated round that barely progresses halts early (saturation)."""
    a = Document(content="aaa", score=0.9)
    engine = _rerank_engine([a], [(a, 0.95)], reply="unused")
    # Always length-truncated but only a sliver of text → context saturated.
    engine.generator.run.return_value = {
        "replies": ["x"],
        "meta": [{"finish_reason": "length"}],
    }
    finish_reasons = []

    answer, _ = run_query_reranked(
        "q",
        engine=engine,
        settings=_settings(max_continuation_rounds=5),
        finish_reason_callback=finish_reasons.append,
    )

    # Initial + ONE continuation, then the non-progress guard stops it (not 5).
    assert engine.generator.run.call_count == 2
    assert answer == "xx"
    assert finish_reasons == ["length"]


def test_continue_reranked_answer_uses_grounded_sources_only() -> None:
    used = Document(content="grounded", score=0.9)
    unused = Document(content="not grounded", score=0.8)
    sources = [
        RankedSource(used, 1, 0.9, 1, 0.95, used_for_grounding=True),
        RankedSource(unused, 2, 0.8, 2, 0.50, used_for_grounding=False),
    ]
    engine = _rerank_engine([], [])
    engine.generator.run.return_value = {
        "replies": [" completion"],
        "meta": [{"finish_reason": "stop"}],
    }
    finish_reasons = []

    continuation = continue_reranked_answer(
        "q",
        "partial",
        sources,
        engine=engine,
        finish_reason_callback=finish_reasons.append,
    )

    prompt = engine.generator.run.call_args.kwargs["prompt"]
    assert continuation == " completion"
    assert "grounded" in prompt
    assert "not grounded" not in prompt
    assert "partial" in prompt
    assert finish_reasons == ["stop"]


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
    # Nothing was fed to the generator, so nothing is marked grounding.
    assert not any(s.used_for_grounding for s in sources)
    engine.generator.run.assert_not_called()


# --- live ----------------------------------------------------------------


def _live_engine_and_store():
    from corpus_rag.document_store import build_document_store
    from corpus_rag.pipelines.query import build_query_engine
    from corpus_rag.settings import get_settings

    settings = get_settings()
    store = build_document_store(settings)
    if store.count_documents() == 0:
        pytest.skip("Empty corpus; ingest a sample corpus first (live).")
    return build_query_engine(store, settings), store, settings


def _corpus_answerable_query(store, settings) -> str:
    """A question the *ingested corpus* can actually answer (domain-agnostic).

    Corpus scope is set at runtime, so a hardcoded clinical query (e.g. about
    pneumonia) legitimately abstains on a corpus that doesn't cover it — which
    would falsely fail the grounded-path test. Instead derive a question from a
    real chunk via the eval auto-qrels machinery, guaranteeing the answer exists
    in-corpus so the grounded path (not abstention) is exercised.
    """
    from haystack.components.generators import OpenAIGenerator
    from haystack.utils import Secret

    from corpus_rag.eval.harness import auto_generate_qrels

    generator = OpenAIGenerator(
        api_key=Secret.from_token("not-needed-for-local-server"),
        model=settings.llm_model,
        api_base_url=settings.llm_base_url,
        generation_kwargs={"temperature": 0},
        timeout=settings.llm_timeout,
    )

    def generate(prompt: str) -> str:
        replies = generator.run(prompt=prompt).get("replies") or []
        return replies[0] if replies else ""

    cases = auto_generate_qrels(store.filter_documents(), generate, n=1)
    if not cases:
        pytest.skip("Could not derive a corpus-answerable query.")
    return cases[0].query


@pytest.mark.live
def test_live_retrieval_count_and_ordering() -> None:
    """§7.5: count bounded by min(TOP_K, N), non-increasing scores, floor honored.

    Works under any MIN_SCORE (no skip): the gate only drops docs, so the count
    is always ≤ min(TOP_K, N); with the floor off it is exactly that.
    """
    engine, store, settings = _live_engine_and_store()
    # A corpus-answerable query so ≥1 doc survives the floor (domain-agnostic).
    query = _corpus_answerable_query(store, settings)
    _, docs = run_query(query, engine=engine, settings=settings)

    expected_max = min(settings.top_k, store.count_documents())
    assert 1 <= len(docs) <= expected_max
    scores = [d.score for d in docs]
    assert all(a >= b for a, b in zip(scores, scores[1:], strict=False))
    if settings.min_score > 0.0:
        # Every surfaced doc cleared the grounding floor.
        assert all((d.score or 0.0) >= settings.min_score for d in docs)
    else:
        # Floor off: the retriever returns the full top-k (no post-filtering).
        assert len(docs) == expected_max


@pytest.mark.live
def test_live_retrieval_order_is_deterministic() -> None:
    """§7.7: same query + corpus yields a stable retrieval order."""
    engine, _, settings = _live_engine_and_store()
    q = "What does the guideline recommend?"
    _, first = run_query(q, engine=engine, settings=settings)
    _, second = run_query(q, engine=engine, settings=settings)

    assert [d.id for d in first] == [d.id for d in second]


@pytest.mark.live
def test_live_a2_non_abstain_answer_has_verbatim_source() -> None:
    """§2A A2: a grounded answer is accompanied by >=1 verbatim source chunk."""
    engine, store, settings = _live_engine_and_store()
    query = _corpus_answerable_query(store, settings)
    answer, docs = run_query(query, engine=engine, settings=settings)

    assert answer != ABSTENTION_ANSWER
    assert len(docs) >= 1
    # Retrieved content is the stored verbatim chunk (same field), byte-equal.
    stored = {d.id: d.content for d in store.filter_documents()}
    assert docs[0].content == stored[docs[0].id]
    assert docs[0].content.strip()


@pytest.mark.live
def test_live_a1_no_match_abstains() -> None:
    """§2A A1: with no grounding above the floor, abstain (no parametric claim)."""
    engine, _, settings = _live_engine_and_store()
    # Force the no-grounding path via a high MIN_SCORE floor.
    strict = settings.model_copy(update={"min_score": 0.999})
    answer, _ = run_query("zzzz unrelated nonsense qqqq", engine=engine, settings=strict)

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
        _corpus_answerable_query(store, settings), engine=engine, settings=settings
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
    engine, store, settings = _live_rerank_engine_and_store()
    # Domain-agnostic: derive a query the corpus answers (same as the other live
    # grounded tests) instead of a hardcoded clinical query that would retrieve
    # unrelated chunks — and randomly (not) reorder — on a non-clinical corpus.
    query = _corpus_answerable_query(store, settings)
    _, sources = run_query_reranked(query, engine=engine, settings=settings)

    if len(sources) < 5:
        pytest.skip("Too few sources to expect a rerank/cosine disagreement.")
    # If rerank changed nothing, cosine_rank would equal rerank_rank for all.
    cosine_order = [s.cosine_rank for s in sources]
    assert cosine_order != sorted(cosine_order), "rerank did not reorder cosine list"
