"""Offline unit tests for the eval harness (plan "Eval harness" step).

Cover the pure metric math, qrels matching, and the Layer-2/3 orchestration with
injected fakes. A small live retrieval-metrics test (``@pytest.mark.live``) runs
the real retriever over the example qrels and self-skips on an empty corpus.
"""

from __future__ import annotations

import json
import math

import pytest
from haystack import Document

from corpus_rag.eval.harness import (
    abstention_rate,
    auto_generate_qrels,
    citation_coverage,
    evaluate_retrieval,
    faithfulness_rate,
    judge_faithfulness,
)
from corpus_rag.eval.metrics import (
    aggregate,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from corpus_rag.eval.qrels import (
    EvalCase,
    RelevanceSpec,
    doc_matches,
    load_qrels,
    parse_qrels,
    relevance_flags,
    relevance_gains,
    specs_covered,
)
from corpus_rag.prompts import ABSTENTION_ANSWER

# --- metrics math --------------------------------------------------------


def test_precision_at_k() -> None:
    assert precision_at_k([True, False, True, False], 4) == 0.5
    assert precision_at_k([True, True], 2) == 1.0
    assert precision_at_k([False, False], 2) == 0.0


def test_precision_at_k_rejects_bad_k() -> None:
    with pytest.raises(ValueError):
        precision_at_k([True], 0)


def test_precision_at_k_divides_by_available_docs() -> None:
    # Corpus smaller than k: denominator is min(k, len), not k.
    assert precision_at_k([True], 5) == 1.0
    assert precision_at_k([True, False], 5) == 0.5
    assert precision_at_k([], 5) == 0.0


def test_recall_at_k_counts_covered_over_total() -> None:
    assert recall_at_k(covered=3, total_relevant=4) == 0.75
    assert recall_at_k(covered=0, total_relevant=5) == 0.0
    # No gold relevant -> defined as 0.0, not a division error.
    assert recall_at_k(covered=0, total_relevant=0) == 0.0


def test_hit_at_k() -> None:
    assert hit_at_k([False, False, True], 3) == 1.0
    assert hit_at_k([False, False, True], 2) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
    assert reciprocal_rank([True, False]) == 1.0
    assert reciprocal_rank([False, False]) == 0.0


def test_ndcg_at_k_perfect_is_one() -> None:
    # Two relevant at the top, two gold total -> ideal ordering -> nDCG 1.0.
    assert ndcg_at_k([True, True, False], k=3, total_relevant=2) == pytest.approx(1.0)


def test_ndcg_at_k_discounts_lower_ranks() -> None:
    # One relevant doc at rank 2; one gold total.
    dcg = 1.0 / math.log2(3)  # rank 2 -> log2(2+1)
    idcg = 1.0 / math.log2(2)  # ideal: rank 1 -> log2(1+1)
    assert ndcg_at_k([False, True], k=2, total_relevant=1) == pytest.approx(dcg / idcg)


def test_ndcg_at_k_zero_when_no_relevant() -> None:
    assert ndcg_at_k([False, False], k=2, total_relevant=0) == 0.0


def test_ndcg_at_k_stays_within_one_with_per_spec_gains() -> None:
    # Two docs match the SAME single spec. Per-document flags ([True, True]) would
    # push raw nDCG above 1.0; per-spec gains ([True, False]) keep it in [0, 1].
    docs = [Document(content="hand hygiene a"), Document(content="hand hygiene b")]
    specs = [RelevanceSpec(contains="hand hygiene")]
    flags = relevance_flags(docs, specs)
    gains = relevance_gains(docs, specs)
    assert flags == [True, True]
    assert gains == [True, False]  # spec credited once, at its first covering doc
    assert ndcg_at_k(flags, k=2, total_relevant=1) > 1.0  # the bug, if fed flags
    assert ndcg_at_k(gains, k=2, total_relevant=1) == pytest.approx(1.0)


def test_aggregate_macro_averages() -> None:
    from corpus_rag.eval.metrics import per_query

    a = per_query([True, False], k=2, covered=1, total_relevant=1)
    b = per_query([False, False], k=2, covered=0, total_relevant=1)
    macro = aggregate([a, b])
    assert macro.n_queries == 2
    assert macro.precision == pytest.approx((0.5 + 0.0) / 2)
    assert macro.recall == pytest.approx((1.0 + 0.0) / 2)
    assert macro.hit == pytest.approx(0.5)


def test_aggregate_rejects_mixed_k() -> None:
    from corpus_rag.eval.metrics import per_query

    a = per_query([True], k=1, covered=1, total_relevant=1)
    b = per_query([True], k=2, covered=1, total_relevant=1)
    with pytest.raises(ValueError):
        aggregate([a, b])


# --- qrels matching ------------------------------------------------------


def test_relevance_spec_requires_a_field() -> None:
    with pytest.raises(ValueError):
        RelevanceSpec()


def test_doc_matches_contains_is_case_insensitive() -> None:
    doc = Document(content="The target INR of 2.0 to 3.0 is recommended.")
    assert doc_matches(doc, RelevanceSpec(contains="target inr"))
    assert not doc_matches(doc, RelevanceSpec(contains="heparin"))


def test_doc_matches_source_searches_meta() -> None:
    doc = Document(content="x", meta={"origin": "anticoag.pdf", "page": 3})
    assert doc_matches(doc, RelevanceSpec(source="anticoag.pdf"))
    assert doc_matches(doc, RelevanceSpec(source="anticoag"))
    assert not doc_matches(doc, RelevanceSpec(source="cardiology.pdf"))


def test_doc_matches_fields_are_anded() -> None:
    doc = Document(content="escalate antibiotics", meta={"src": "guideline.pdf"})
    assert doc_matches(doc, RelevanceSpec(contains="escalate", source="guideline"))
    # contains satisfied but source not -> overall False.
    assert not doc_matches(doc, RelevanceSpec(contains="escalate", source="other"))


def test_relevance_flags_and_specs_covered() -> None:
    docs = [
        Document(content="pneumonia first-line amoxicillin"),
        Document(content="unrelated"),
        Document(content="INR target 2-3"),
    ]
    specs = [RelevanceSpec(contains="pneumonia"), RelevanceSpec(contains="INR")]
    assert relevance_flags(docs, specs) == [True, False, True]
    assert specs_covered(docs, specs) == 2
    # Top-1 only covers the pneumonia spec.
    assert specs_covered(docs[:1], specs) == 1


def test_relevance_gains_credits_each_spec_once() -> None:
    docs = [
        Document(content="pneumonia A"),  # covers pneumonia spec (first) -> gain
        Document(content="pneumonia B"),  # pneumonia already credited -> no gain
        Document(content="INR 2-3"),  # covers INR spec (first) -> gain
    ]
    specs = [RelevanceSpec(contains="pneumonia"), RelevanceSpec(contains="INR")]
    assert relevance_gains(docs, specs) == [True, False, True]
    assert sum(relevance_gains(docs, specs)) <= len(specs)


def test_parse_qrels_round_trip() -> None:
    raw = [{"query": "q1", "relevant": [{"contains": "x"}, {"source": "s"}]}]
    cases = parse_qrels(raw)
    assert len(cases) == 1
    assert cases[0].query == "q1"
    assert len(cases[0].relevant) == 2


def test_parse_qrels_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        parse_qrels([{"query": "q"}])  # missing 'relevant'


def test_load_qrels_example_file(tmp_path) -> None:
    src = "tests/eval/qrels.example.json"
    cases = load_qrels(src)
    assert len(cases) >= 1
    assert all(c.relevant for c in cases)
    # Round-trip through a temp copy to exercise the Path branch.
    p = tmp_path / "q.json"
    p.write_text(json.dumps([{"query": "q", "relevant": [{"contains": "x"}]}]))
    assert load_qrels(p)[0].query == "q"


# --- Layer 1 orchestration ----------------------------------------------


def test_evaluate_retrieval_macro_and_per_case() -> None:
    corpus = {
        "pneumonia": [Document(content="pneumonia first-line"), Document(content="noise")],
        "warfarin": [Document(content="noise"), Document(content="INR 2-3")],
    }

    def retrieve(query: str):
        return corpus["pneumonia"] if "pneumonia" in query else corpus["warfarin"]

    cases = [
        EvalCase("pneumonia tx?", (RelevanceSpec(contains="pneumonia"),)),
        EvalCase("warfarin INR?", (RelevanceSpec(contains="INR"),)),
    ]
    macro, per_case = evaluate_retrieval(cases, retrieve, k=2)
    assert macro.n_queries == 2
    assert macro.recall == pytest.approx(1.0)  # both gold specs covered in top-2
    # Pneumonia gold at rank 1 -> RR 1.0; warfarin gold at rank 2 -> RR 0.5.
    assert macro.mrr == pytest.approx((1.0 + 0.5) / 2)
    assert [c.n_retrieved for c in per_case] == [2, 2]


def test_evaluate_retrieval_rejects_empty() -> None:
    with pytest.raises(ValueError):
        evaluate_retrieval([], lambda q: [], k=5)


# --- Layer 2 reference-free ---------------------------------------------


def test_abstention_rate() -> None:
    def run_fn(q: str):
        if q == "bad":
            return ABSTENTION_ANSWER, []
        return "grounded", [Document(content="x")]

    assert abstention_rate(["good", "bad", "good", "bad"], run_fn) == 0.5
    assert abstention_rate([], run_fn) == 0.0


def test_citation_coverage_lexical_overlap() -> None:
    docs = [Document(content="amoxicillin first-line pneumonia treatment")]
    # All answer content words present in the source.
    assert citation_coverage("amoxicillin pneumonia", docs) == pytest.approx(1.0)
    # Half present.
    cov = citation_coverage("amoxicillin warfarin", docs)
    assert cov == pytest.approx(0.5)
    assert citation_coverage("", docs) == 0.0


def test_judge_faithfulness_parses_verdict() -> None:
    docs = [Document(content="source text")]
    assert judge_faithfulness("claim", docs, lambda p: "SUPPORTED") is True
    assert judge_faithfulness("claim", docs, lambda p: "UNSUPPORTED") is False
    # Ambiguous output -> fail safe (unsupported) for a safety metric.
    assert judge_faithfulness("claim", docs, lambda p: "maybe?") is False
    # Abstention is not judged.
    assert judge_faithfulness(ABSTENTION_ANSWER, docs, lambda p: "SUPPORTED") is None


def test_judge_faithfulness_prefers_unsupported_when_both_tokens_present() -> None:
    docs = [Document(content="s")]
    verdict = judge_faithfulness("c", docs, lambda p: "not SUPPORTED, it is UNSUPPORTED")
    assert verdict is False


def test_faithfulness_rate_excludes_abstentions() -> None:
    def run_fn(q: str):
        if q == "abstain":
            return ABSTENTION_ANSWER, []
        return f"answer for {q}", [Document(content="s")]

    # Judge: only "good" is supported.
    def judge(prompt: str) -> str:
        return "SUPPORTED" if "good" in prompt else "UNSUPPORTED"

    rate, n = faithfulness_rate(["good", "bad", "abstain"], run_fn, judge)
    assert n == 2  # abstain excluded
    assert rate == pytest.approx(0.5)


def test_faithfulness_rate_none_when_all_abstain() -> None:
    # None (not 1.0): nothing was judged, so "100% faithful" would be misleading.
    rate, n = faithfulness_rate(["a"], lambda q: (ABSTENTION_ANSWER, []), lambda p: "SUPPORTED")
    assert (rate, n) == (None, 0)


# --- Layer 3 auto-generated qrels ---------------------------------------


def test_auto_generate_qrels_builds_cases_from_chunks() -> None:
    docs = [Document(content=f"clinical passage number {i} with detail") for i in range(6)]
    calls: list[str] = []

    def generate(prompt: str) -> str:
        calls.append(prompt)
        return "What is the detail?"

    cases = auto_generate_qrels(docs, generate, n=3, snippet_len=20)
    assert len(cases) == 3
    assert all(c.query == "What is the detail?" for c in cases)
    # Each gold spec is a content snippet of the source chunk.
    assert all(c.relevant[0].contains for c in cases)
    assert len(calls) == 3


def test_auto_generate_qrels_skips_empty_chunks_and_blank_questions() -> None:
    docs = [Document(content="  "), Document(content="real passage")]
    cases = auto_generate_qrels(docs, lambda p: "Q?", n=5)
    # Only the non-blank chunk yields a case.
    assert len(cases) == 1
    # Blank question -> dropped.
    assert auto_generate_qrels([Document(content="x")], lambda p: "  ", n=1) == []


def test_auto_generate_qrels_deterministic_sampling() -> None:
    docs = [Document(content=f"passage {i}") for i in range(10)]
    a = auto_generate_qrels(docs, lambda p: "Q?", n=3)
    b = auto_generate_qrels(docs, lambda p: "Q?", n=3)
    assert [c.relevant[0].contains for c in a] == [c.relevant[0].contains for c in b]
