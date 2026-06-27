"""Eval harness for the Corpus RAG Explorer (plan "Eval harness" step).

OPTIONAL, beyond-MVP — root ``spec.md`` §8 defers an eval harness. Added per
user request. Because corpus scope is set at RUNTIME (the deployer uploads the
domain corpus), this harness builds MACHINERY + corpus-relative metrics and
never hardcodes domain truth. Three layers, by who owns the ground truth:

- **Layer 1 — labelled retrieval metrics** (``metrics`` + ``qrels`` + harness):
  precision@k / recall@k / MRR / nDCG@k / hit@k against a gold qrels set. Used at
  dev time over a fixed fixture corpus (regression gate) AND, with Layer 3, over
  the runtime corpus.
- **Layer 2 — reference-free metrics** (``harness``): grounding faithfulness
  (LLM-judge: are the answer's specific claims entailed by the retrieved chunks?),
  citation coverage, abstention rate. No labels; truth = the uploaded corpus.
- **Layer 3 — auto-generated qrels** (``harness.auto_generate_qrels``): an LLM
  reads a chunk and writes a question whose answer is that chunk → that chunk is
  the gold-relevant doc. Yields recall@k / MRR on the actual deployed corpus with
  no human labelling.

The pure-math (``metrics``) and matching (``qrels``) layers are import-only and
fully offline-testable. The LLM-driven Layer-2/3 functions take injectable
callables so they can be unit-tested with fakes.
"""

from __future__ import annotations

from corpus_rag.eval.metrics import (
    RetrievalMetrics,
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
    relevance_flags,
    relevance_gains,
    specs_covered,
)

__all__ = [
    "EvalCase",
    "RelevanceSpec",
    "RetrievalMetrics",
    "aggregate",
    "doc_matches",
    "hit_at_k",
    "load_qrels",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "relevance_flags",
    "relevance_gains",
    "specs_covered",
]
