"""Eval harness CLI (plan "Eval harness" step) — runs against LIVE services.

OPTIONAL, beyond-MVP (root ``spec.md`` §8 defers eval). Corpus scope is set at
runtime, so this never hardcodes domain truth: it scores retrieval against a gold
qrels set (Layer 1), or auto-generates that set from the ingested corpus (Layer
3), and optionally runs reference-free grounding metrics (Layer 2).

Run:
  # Layer 1: score against a hand-authored gold set
  uv run python scripts/eval.py --qrels tests/eval/qrels.json

  # Layer 3: auto-generate qrels from the live corpus, then score
  uv run python scripts/eval.py --auto-generate 25

  # add Layer 2 reference-free grounding metrics over the same queries
  uv run python scripts/eval.py --qrels tests/eval/qrels.json --reference-free
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pgvector import (
    PgvectorEmbeddingRetriever,
)

from corpus_rag.document_store import build_document_store
from corpus_rag.eval.harness import (
    abstention_rate,
    auto_generate_qrels,
    citation_coverage,
    evaluate_retrieval,
    faithfulness_rate,
)
from corpus_rag.eval.qrels import load_qrels
from corpus_rag.pipelines.query import run_query
from corpus_rag.prompts import ABSTENTION_ANSWER
from corpus_rag.settings import get_settings


def _build_retrieve_fn(store, settings, *, k: int):
    """A warmed embedder+retriever returning top-``k`` docs in cosine order."""
    embedder = SentenceTransformersTextEmbedder(model=settings.embed_model_id)
    embedder.warm_up()
    retriever = PgvectorEmbeddingRetriever(document_store=store, top_k=k)

    def retrieve(query: str):
        embedding = embedder.run(text=query)["embedding"]
        return retriever.run(query_embedding=embedding)["documents"]

    return retrieve


def _build_generate_fn(settings):
    """A single-shot LLM call (temperature 0) for the judge / question-gen."""
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

    return generate


def _fmt_metrics_table(macro) -> list[str]:
    row = macro.as_row()
    return [
        "| Metric | Value |",
        "| --- | --- |",
        f"| queries | {row['queries']} |",
        f"| precision@{row['k']} | {row['precision@k']} |",
        f"| recall@{row['k']} | {row['recall@k']} |",
        f"| MRR | {row['mrr']} |",
        f"| nDCG@{row['k']} | {row['ndcg@k']} |",
        f"| hit@{row['k']} | {row['hit@k']} |",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval + grounding eval harness.")
    parser.add_argument("--qrels", default=None, help="Path to a JSON qrels file (Layer 1).")
    parser.add_argument(
        "--auto-generate",
        type=int,
        default=0,
        metavar="N",
        help="Auto-generate N qrels from the live corpus (Layer 3) instead of --qrels.",
    )
    parser.add_argument("--k", type=int, default=None, help="Cutoff k (default: TOP_K).")
    parser.add_argument(
        "--reference-free",
        action="store_true",
        help="Also run Layer-2 reference-free metrics (abstention, faithfulness, citation).",
    )
    parser.add_argument("--report", default=None, help="Write a Markdown report to this path.")
    args = parser.parse_args()

    settings = get_settings()
    k = args.k or settings.top_k
    store = build_document_store(settings)
    if store.count_documents() == 0:
        print("Empty corpus; ingest a corpus first (uv run corpus-rag ingest).", file=sys.stderr)
        return 2

    retrieve_fn = _build_retrieve_fn(store, settings, k=k)

    # --- assemble eval cases (Layer 1 gold set OR Layer 3 auto-generated) ---
    generate_fn = None  # built lazily; reused by Layer 3 and Layer 2
    if args.auto_generate > 0:
        print(f"Auto-generating {args.auto_generate} qrels from the corpus (Layer 3)...")
        generate_fn = _build_generate_fn(settings)
        cases = auto_generate_qrels(store.filter_documents(), generate_fn, n=args.auto_generate)
        source_desc = f"auto-generated ({len(cases)} from corpus)"
    else:
        qrels_path = args.qrels or "tests/eval/qrels.json"
        if not Path(qrels_path).exists():
            print(
                f"qrels file not found: {qrels_path}\n"
                "Provide --qrels PATH or use --auto-generate N.",
                file=sys.stderr,
            )
            return 2
        cases = load_qrels(qrels_path)
        source_desc = f"{qrels_path} ({len(cases)} cases)"

    if not cases:
        print("No eval cases to run.", file=sys.stderr)
        return 2

    # --- Layer 1: retrieval metrics ---
    macro, per_case = evaluate_retrieval(cases, retrieve_fn, k=k)
    print(f"\nLayer 1 — retrieval metrics ({source_desc}):")
    for line in _fmt_metrics_table(macro):
        print(line)

    report_lines = [
        "# Corpus RAG Explorer — Eval Report",
        "",
        f"**Eval set:** {source_desc}",
        f"**Cutoff k:** {k}",
        "",
        "## Layer 1 — retrieval metrics (macro-average)",
        "",
        *_fmt_metrics_table(macro),
        "",
        "### Per-query",
        "",
        "| Query | recall@k | MRR | retrieved | covered/total |",
        "| --- | --- | --- | --- | --- |",
    ]
    total_by_query = {c.query: len(c.relevant) for c in cases}
    for c in per_case:
        q = c.query if len(c.query) <= 60 else c.query[:57] + "..."
        report_lines.append(
            f"| {q} | {round(c.metrics.recall, 3)} | {round(c.metrics.mrr, 3)} "
            f"| {c.n_retrieved} | {c.covered}/{total_by_query.get(c.query, 0)} |"
        )

    # --- Layer 2: reference-free metrics (optional) ---
    if args.reference_free:
        print("\nLayer 2 — reference-free grounding metrics...")
        if generate_fn is None:
            generate_fn = _build_generate_fn(settings)
        queries = [c.query for c in cases]

        def run_fn(q: str):
            return run_query(q, settings=settings)

        abstain = abstention_rate(queries, run_fn)
        faith, n_judged = faithfulness_rate(queries, run_fn, generate_fn)
        faith_str = "n/a (all abstained)" if faith is None else round(faith, 3)
        # Citation coverage over non-abstaining answers.
        coverages = []
        for q in queries:
            ans, docs = run_fn(q)
            if ans != ABSTENTION_ANSWER:
                coverages.append(citation_coverage(ans, docs))
        avg_cov = sum(coverages) / len(coverages) if coverages else 0.0

        l2 = [
            "",
            "## Layer 2 — reference-free grounding metrics",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| abstention rate | {round(abstain, 3)} |",
            f"| faithfulness rate (LLM-judge) | {faith_str} (n={n_judged}) |",
            f"| citation coverage (lexical proxy) | {round(avg_cov, 3)} |",
        ]
        for line in l2:
            print(line)
        report_lines += l2

    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        print(f"\nReport written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
